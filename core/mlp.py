import math
import pickle
from pathlib import Path
import threading

import torch
from torch.utils.data import DataLoader

from core.utils import thread_killer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_iterable(val: str, element_type=int) -> list:
    val = val.strip("[](){} ")
    out = []
    for el in val.split(','):
        out.append(element_type(el))
    return out


class MLPSpatial(torch.nn.Module):
    def __init__(self, hidden_layers, norm, activation, pool, n_classes, spatial_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for layer_dim in hidden_layers:
            self.layers.append(torch.nn.LazyConv2d(layer_dim, 1))
            self.layers.append(pool(2, 2, ceil_mode=True))
            spatial_dim = math.ceil(spatial_dim / 2)
            self.layers.append(norm())
            self.layers.append(activation())
        self.layers.append(pool(spatial_dim, 1))
        self.layers.append(torch.nn.LazyConv2d(n_classes, 1, bias=False))

    def forward(self, x, dropout=False):
        dropout = dropout if isinstance(dropout, float) else 0.2
        for module in self.layers:
            x = module(x)
            if dropout and isinstance(module, (torch.nn.Conv2d, torch.nn.LazyConv2d)):
                x = torch.nn.Dropout(dropout)(x)
        return x

    @torch.no_grad()
    def get_activations(self, x, layer=-1):
        all_activations = []
        for module in self.layers:
            x = module(x)
            if isinstance(module, (torch.nn.Conv2d, torch.nn.LazyConv2d)):
                all_activations.append(x.permute(0, 2, 3, 1).reshape(x.shape[0], -1))
        return all_activations[layer]


class MLPPooled(torch.nn.Module):
    def __init__(self, hidden_layers, norm, activation, n_classes):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for layer_dim in hidden_layers:
            self.layers.append(torch.nn.LazyConv2d(layer_dim, 1))
            self.layers.append(norm())
            self.layers.append(activation())
        self.layers.append(torch.nn.LazyConv2d(n_classes, 1, bias=False))

    def forward(self, x, dropout=False):
        dropout = dropout if isinstance(dropout, float) else 0.2
        for module in self.layers:
            x = module(x)
            if dropout and isinstance(module, (torch.nn.Conv2d, torch.nn.LazyConv2d)):
                x = torch.nn.Dropout(dropout)(x)
        return x

    @torch.no_grad()
    def get_activations(self, x, layer=-1):
        all_activations = []
        for module in self.layers:
            x = module(x)
            if isinstance(module, (torch.nn.Conv2d, torch.nn.LazyConv2d)):
                all_activations.append(x.reshape(x.shape[0], -1))
        return all_activations[layer]


class MLP:
    def __init__(self, hidden_layers="(24, 5)",
                 norm="batch",
                 activation="mish",
                 pool="avg",
                 l2_weight=0.0,
                 learning_rate=0.01,
                 epochs=1000,
                 batch_size=2048,
                 device=DEVICE,
                 use_spatial=False,
                 augs=False,
                 ):
        """

        :param hidden_layers: iterable of hidden layer sizes. the head is attached during fit
        :param norm: type of normalization in ["identity", "batch", "layer"]
        :param activation: type of activation in ["identity", "tanh", "mish"]
        :param pool: type of spatial pooling in ["avg", "max"]
        :param l2_weight: weight of l2 regularization in loss
        :param learning_rate:
        :param epochs: number of epochs
        :param batch_size: use -1 to optimize whole dataset
        """
        hidden_layers = parse_iterable(hidden_layers)
        assert len(hidden_layers), f"Provide at least one hidden layer, received {hidden_layers}"
        self.hidden_layers = hidden_layers
        self.norm = norm
        self.activation = activation
        self.pool = pool
        self.l2_weight = l2_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.use_spatial = use_spatial
        self.augs = augs

        self._categories = None
        self.model: torch.nn.Module = None
        self.callback = print

    def _init(self, dataset):
        _activations_lookup = {
            "identity": torch.nn.Identity,
            "mish": torch.nn.Mish,
            "tanh": torch.nn.Tanh,
        }
        _norm_lookup = {
            "identity": torch.nn.Identity,
            "batch": torch.nn.LazyBatchNorm2d,
            # "layer": torch.nn.LayerNorm,
        }
        _pool_lookup = {
            "avg": torch.nn.AvgPool2d,
            "max": torch.nn.MaxPool2d,
        }
        n_classes = len(self._categories)
        spatial_dim = dataset[0][0].shape[-1]
        if n_classes < 3:
            n_classes -= 1
        assert n_classes > 0, f"Wrong class number, received {n_classes}"
        if self.use_spatial:
            self.model = MLPSpatial(self.hidden_layers, _norm_lookup[self.norm],
                                    _activations_lookup[self.activation],
                                    _pool_lookup[self.pool],
                                    n_classes,
                                    spatial_dim)
        else:
            self.model = MLPPooled(self.hidden_layers, _norm_lookup[self.norm],
                                   _activations_lookup[self.activation],
                                   n_classes)

    def save(self, save_path):
        pt_path = Path(save_path) / "model.pt"
        torch.save(self.model, pt_path)
        model = self.model
        self.model = None
        callback = self.callback
        self.callback = None
        with open(save_path / "model.pkl", "wb") as file:
            pickle.dump(self, file)
        self.model = model
        self.callback = callback

    @classmethod
    def load(cls, file_path):
        with open(file_path / "model.pkl", "rb") as file:
            obj = pickle.load(file)

        pt_path = file_path / "model.pt"
        model = None
        if pt_path.exists():
            model = torch.load(pt_path, map_location=obj.device)

        obj.model = model
        return obj

    def fit(self, X, y, test_func=None):
        self._categories = sorted(set(y))
        self._init(X)
        self.model = self.model.to(self.device)
        self.model.train()
        loss_func = torch.nn.BCEWithLogitsLoss() if len(self._categories) == 2 else torch.nn.CrossEntropyLoss()
        dataloader = DataLoader(X, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.l2_weight
                                      )
        lr_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, total_iters=self.epochs,
                                                     end_factor=0.001)
        for epoch in range(self.epochs):
            should_log = epoch % max(1, (self.epochs // 30)) == 0
            if should_log:
                self.callback(f"Epoch {epoch}/{self.epochs}", end=" ")
            losses = []
            for batch in dataloader:
                if thread_killer.is_set(threading.current_thread().ident):
                    raise Exception("Canceled")
                data, labels = batch
                preds = self.model(data.to(self.device))
                loss = loss_func(preds.squeeze(), labels.to(self.device))
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            lr_sched.step()
            if should_log:
                self.callback(f"Loss: {sum(losses) / len(losses)}")
                self.model.eval()
                test_func()
                self.model.train()
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, X, dropout=0.2):
        dataloader = DataLoader(X, batch_size=self.batch_size)
        all_logits = []
        for batch in dataloader:
            if thread_killer.is_set(threading.current_thread().ident):
                raise Exception("Canceled")
            data = batch
            preds = self.model(data.to(self.device), dropout=dropout)
            all_logits.append(preds)
        all_logits = torch.cat(all_logits, dim=0)
        if len(all_logits.shape) == 0 or all_logits.shape[-1] == 1:
            return torch.sigmoid(all_logits).flatten()
        else:
            assert len(all_logits.shape) == 2
            return torch.softmax(all_logits, dim=-1)

    @torch.no_grad()
    def predict(self, X, probas=None):
        if probas is None:
            probas = self.predict_proba(X)
        if len(probas.shape) > 1:
            return torch.argmax(probas, dim=-1).cpu()
        else:
            return torch.where(probas > 0.5, 1, 0).cpu()

    @torch.no_grad()
    def get_activations(self, X, layer=-1):
        dataloader = DataLoader(X, batch_size=self.batch_size)
        all_activations = []
        for batch in dataloader:
            if thread_killer.is_set(threading.current_thread().ident):
                raise Exception("Canceled")
            data = batch
            all_activations.append(self.model.get_activations(data.to(self.device), layer=layer))
        return torch.cat(all_activations, dim=0).cpu().numpy()

    @torch.no_grad()
    def get_importance(self, X, dropout_runs=16):
        runs = []
        for i in range(dropout_runs):
            self.callback(f"Dropout run #{i}")
            runs.append(self.predict_proba(X))

        runs = torch.stack(runs, dim=0)
        point_estimates = runs.mean(dim=0)
        if len(point_estimates.shape) < 2:
            uncertainty = -torch.log(torch.maximum(point_estimates, 1 - point_estimates))
        else:
            uncertainty = -torch.log(torch.max(point_estimates, dim=-1)[0])

        score = uncertainty

        return point_estimates, score
