import math
import pickle
from pathlib import Path
import threading
from itertools import zip_longest

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.cluster import BisectingKMeans

from core.utils import thread_killer, TrainDataset

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

    def forward(self, x, dropout=0.2):
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

    def forward(self, x, dropout=0.2):
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
                 importance_mode="uncertainty",
                 importance_sort="coverage"
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
        :param importance_mode: str in ["uncertainty", "positive", "negative"]
        :param importance_sort: str in ["simple", "coverage"]. coverage uses bisect kmeans to prioritize diversity
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
        self.importance_mode = importance_mode
        self.importance_sort = importance_sort

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

    def fit(self, X: TrainDataset, y, test_func=None):
        self._categories = sorted(set(y))
        self._init(X)
        self.model = self.model.to(self.device)
        self.model.train()
        loss_func = torch.nn.BCEWithLogitsLoss() if len(self._categories) == 2 else torch.nn.CrossEntropyLoss()

        def train_metrics_func(_logits, _labels):
            _preds = self.predict(None, probas=self.get_proba(_logits))
            return {"f1_score": f1_score(_labels.cpu().numpy(), _preds.detach().cpu().numpy())}

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
            train_metrics = []
            for batch in dataloader:
                if thread_killer.is_set(threading.current_thread().ident):
                    raise Exception("Canceled")
                data, labels = batch
                preds = self.model(data.to(self.device))
                loss = loss_func(preds.squeeze(), labels.to(self.device))
                loss.backward()
                losses.append(loss.item())
                train_metrics.append(train_metrics_func(preds, labels))
                optimizer.step()
            lr_sched.step()
            if should_log:
                self.callback(f"Loss: {sum(losses) / len(losses)}")
                aggregated_metrics = {k: sum(m[k] for m in train_metrics)/len(train_metrics) for k in train_metrics[0]}
                self.callback(f"Train metrics: {aggregated_metrics}")
                self.model.eval()
                test_func()
                self.model.train()
        self.model.eval()
        X.empty_cache()

    @staticmethod
    def get_proba(logits):
        if len(logits.shape) == 0 or logits.shape[-1] == 1:
            return torch.sigmoid(logits).flatten()
        else:
            assert len(logits.shape) == 2
            return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_proba(self, X, dropout=False):
        dataloader = DataLoader(X, batch_size=self.batch_size)
        all_logits = []
        for batch in dataloader:
            if thread_killer.is_set(threading.current_thread().ident):
                raise Exception("Canceled")
            data = batch
            preds = self.model(data.to(self.device), dropout=dropout)
            all_logits.append(preds)
        all_logits = torch.cat(all_logits, dim=0)
        return self.get_proba(all_logits)

    @torch.no_grad()
    def predict(self, X, probas=None):
        """If probabilities are provided, does thresholding only"""
        if probas is None:
            probas = self.predict_proba(X)
        if len(probas.shape) > 1:
            return torch.argmax(probas, dim=-1).cpu()
        else:
            return torch.where(probas > 0.5, 1, 0).cpu()

    @torch.no_grad()
    def get_activations(self, X, layer=-1):
        """Retrieve the linear layer activations. Id of the layer can be specified"""
        dataloader = DataLoader(X, batch_size=self.batch_size)
        all_activations = []
        for batch in dataloader:
            if thread_killer.is_set(threading.current_thread().ident):
                raise Exception("Canceled")
            data = batch
            all_activations.append(self.model.get_activations(data.to(self.device), layer=layer))
        return torch.cat(all_activations, dim=0).cpu().numpy()

    @torch.no_grad()
    def get_importance(self, X: TrainDataset, dropout_runs=16):
        """
        Estimates predictions with dropout runs, then scores samples by
        their importance and calculates the order
        :param X: input data
        :param dropout_runs: number of times to run inference with dropout
        :return: class probabilities, metadata, and order of importance
        """
        runs = []
        for i in range(dropout_runs):
            self.callback(f"Dropout run #{i}")
            runs.append(self.predict_proba(X, dropout=0.2))

        runs = torch.stack(runs, dim=0)
        point_estimates = runs.mean(dim=0)
        if self.importance_mode == "uncertainty":
            if len(point_estimates.shape) < 2:
                uncertainty = -torch.log(torch.maximum(point_estimates, 1 - point_estimates))
            else:
                uncertainty = -torch.log(torch.max(point_estimates, dim=-1)[0])

            score = uncertainty
        elif self.importance_mode == "positive":
            if len(point_estimates.shape) < 2:
                score = point_estimates
            else:
                score = torch.max(point_estimates, dim=-1)[0]
        elif self.importance_mode == "negative":
            if len(point_estimates.shape) < 2:
                score = 1 - point_estimates
            else:
                raise ValueError(f"Negative class undefined for multiclass task")
        else:
            raise ValueError(f"Invalid importance mode: {self.importance_mode}")

        if self.importance_sort == "coverage":
            activations = self.get_activations(X, layer=-2)
            kmeans = BisectingKMeans(n_clusters=16, init="k-means++", bisecting_strategy="largest_cluster")
            clusters = kmeans.fit_predict(activations).tolist()
            cluster_bins = {}
            for i, c in enumerate(clusters):
                el = cluster_bins.get(c, [])
                el.append(i)
                cluster_bins[c] = el
            for c in cluster_bins:
                cluster_bins[c] = sorted(cluster_bins[c], key=lambda x: score[x], reverse=True)
            order = []
            for slice_ in zip_longest(*cluster_bins.values()):
                order.extend([s for s in slice_ if s is not None])
            metas = [{"importance": score[i].item(),
                      "proba": point_estimates[i].item(),
                      "cluster": clusters[i]}
                     for i in range(len(score))]
        else:
            metas = [{"importance": score[i].item(), "proba": point_estimates[i].item()}
                     for i in range(len(score))]
            order = torch.argsort(score, descending=True).tolist()

        X.empty_cache()
        return point_estimates, metas, order
