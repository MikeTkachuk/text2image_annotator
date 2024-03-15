import json
import time
from pathlib import Path
import pickle
import inspect
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from torch.utils.data import DataLoader, Dataset


def parse_iterable(val: str, element_type=int) -> list:
    val = val.strip("[](){} ")
    out = []
    for el in val.split(','):
        out.append(element_type(el))
    return out


class MLP:
    def __init__(self, hidden_layers="(24, 5)",
                 norm="batch",
                 activation="mish",
                 l2_weight=0.0,
                 learning_rate=0.01,
                 epochs=1000,
                 batch_size=2048,
                 device="cpu"
                 ):
        """

        :param hidden_layers: iterable of hidden layer sizes. the head is attached during fit
        :param norm: type of normalization in ["identity", "batch", "layer"]
        :param activation: type of activation in ["identity", "tanh", "mish"]
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
        self.l2_weight = l2_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self._categories = None
        self.model: torch.nn.Sequential = None
        self.callback = print

    def _init(self):
        _activations_lookup = {
            "identity": torch.nn.Identity,
            "mish": torch.nn.Mish,
            "tanh": torch.nn.Tanh,
        }
        _norm_lookup = {
            "identity": torch.nn.Identity,
            "batch": torch.nn.LazyBatchNorm1d,
            # "layer": torch.nn.LayerNorm,
        }
        modules = torch.nn.ModuleList()
        for layer_dim in self.hidden_layers:
            modules.append(torch.nn.LazyLinear(layer_dim))
            modules.append(_norm_lookup[self.norm]())
            modules.append(_activations_lookup[self.activation]())
        n_classes = len(self._categories)
        if n_classes < 3:
            n_classes -= 1
        assert n_classes > 0, f"Wrong class number, received {n_classes}"
        modules.append(torch.nn.LazyLinear(n_classes, bias=False))
        self.model = torch.nn.Sequential(*modules)

    def save(self, save_path):
        pt_path = Path(save_path) / "model.pt"
        torch.save(self.model, pt_path)
        model = self.model
        self.model = None
        self.callback = None
        with open(save_path / "model.pkl", "wb") as file:
            pickle.dump(self, file)
        self.model = model

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

    @property
    def _dataset(self):
        class _Dataset(Dataset):
            def __init__(self, data, labels=None):
                super().__init__()
                self.data = data
                self.labels = labels
                assert self.labels is None or len(self.data) == len(self.labels)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                if self.labels is not None:
                    return torch.tensor(self.data[idx]).float(), torch.tensor(self.labels[idx]).float()
                else:
                    return torch.tensor(self.data[idx]).float()

        return _Dataset

    # todo: add augs (task registry adds augmented images to set)
    #  (embstore get_image_emb option to store emb variations in one file; regenerate augs option)
    # todo: cross-validation options
    def fit(self, X, y):
        self._categories = sorted(set(y))
        y_id = [self._categories.index(val) for val in y]
        self._init()
        self.model = self.model.to(self.device)
        loss_func = torch.nn.BCEWithLogitsLoss() if len(self._categories) == 2 else torch.nn.CrossEntropyLoss()
        dataset = self._dataset(X, y_id)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.l2_weight
                                      )
        for epoch in range(self.epochs):
            should_log = epoch % 100 == 0
            if should_log:
                self.callback(f"Epoch {epoch}/{self.epochs}", end=" ")
            losses = []
            for batch in dataloader:
                data, labels = batch
                preds = self.model(data.to(self.device))
                loss = loss_func(preds.squeeze(), labels.to(self.device))
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            if should_log:
                self.callback(f"Loss: {sum(losses) / len(losses)}")

    @torch.no_grad()
    def predict_proba(self, X):
        dataset = self._dataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        all_logits = []
        for batch in dataloader:
            data = batch
            preds = self.model(data.to(self.device))
            all_logits.append(preds)
        all_logits = torch.cat(all_logits, dim=0)
        if len(all_logits.shape) == 0 or all_logits.shape[-1] == 1:
            return torch.sigmoid(all_logits).flatten()
        else:
            assert len(all_logits.shape) == 2
            return torch.softmax(all_logits, dim=-1)

    @torch.no_grad()
    def predict(self, X):
        probas = self.predict_proba(X)
        if len(probas.shape) > 1:
            return torch.argmax(probas, dim=-1)
        else:
            return torch.where(probas > 0.5, 1, 0)

    @torch.no_grad()
    def get_activations(self, X, layer=-1):
        dataset = self._dataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        all_activations = []
        for batch in dataloader:
            data = batch
            activations = [self.model[0](data.to(self.device))]
            x = activations[0]
            for i in range(1, len(self.model)):
                x = self.model[i](x)
                if isinstance(self.model[i], (torch.nn.Linear, torch.nn.LazyLinear)):
                    activations.append(x)
            all_activations.append(activations[layer])
        return torch.cat(all_activations, dim=0).cpu().numpy()


class Model:
    def __init__(self, model, params, framework, embstore_name, save_path):
        self.model = model
        self.params = params
        self.framework = framework
        self.save_path = Path(save_path)
        self.embstore_name = embstore_name

        self.last_metrics = {}
        self._model_obj = self.model(**self.params)
        self._predictions: Dict[str, int] = {}

    @staticmethod
    def get_templates():
        return {
            "DecisionTreeClassifier": {"class": DecisionTreeClassifier,
                                       "framework": "sklearn"},
            "RandomForestClassifier": {"class": RandomForestClassifier,
                                       "framework": "sklearn"},
            "LogisticRegression": {"class": LogisticRegression,
                                   "framework": "sklearn"},
            "MLP": {"class": MLP, "framework": "torch"}
        }

    @classmethod
    def from_template(cls, template_name, embstore_name, save_path):
        template = cls.get_templates()[template_name]
        return Model(template["class"], template.get("params", {}), template["framework"], embstore_name, save_path)

    def __dict__(self):
        out = {
            "params": self.params,
            "framework": self.framework,
            "embstore_name": self.embstore_name,
            "predictions": self._predictions,
            "last_metrics": self.last_metrics,
            "template": str(self.model)
        }
        return out

    def save(self):
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        if self.framework == "torch":
            self._model_obj.save(self.save_path)
        else:
            with open(self.save_path / "model.pkl", "wb") as file:
                pickle.dump(self._model_obj, file)

        config_path = self.save_path / "config.json"
        config = {
                  "params": self.params,
                  "framework": self.framework,
                  "embstore_name": self.embstore_name,
                  "predictions": self._predictions,
                  "last_metrics": self.last_metrics
                  }
        with open(config_path, "w") as file:
            json.dump(config, file)

    def get_model_signature(self):
        signature = inspect.signature(self.model.__init__)
        out = {}
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
            default = param.default
            if default == inspect._empty:
                continue
            out[param_name] = default
        return out

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path / "config.json") as file:
            data = json.load(file)
        if data["framework"] == "torch":
            model_obj = MLP.load(path)
        else:
            with open(path / "model.pkl", "rb") as file:
                model_obj = pickle.load(file)
        model = Model(type(model_obj),
                      data["params"],
                      data["framework"],
                      data["embstore_name"],
                      path)
        model._model_obj = model_obj
        model._predictions = data["predictions"]
        model.last_metrics = data["last_metrics"]
        return model

    def _k_fold(self, all_samples, test_samples=None, k=4):
        """

        :param all_samples:
        :param test_samples: if provided, considered as one of the folds and will be yielded last
        :param k:
        :return:
        """

    # todo:
    #  - fit existing samples
    #  - compute validation metrics
    #  - show model emb clusters
    #  - show decision boundary
    #  - suggest based on uncertainty, cluster coverage
    #  - suggestions should be a stream of images sorted by impact/relevance
    def fit(self, dataset: Dict[str, Tuple[Any, int]], test_split=None, callback=None):
        if callback is None:
            callback = print

        callback(f"Model params: {self.params}")
        self._model_obj = self.model(**self.params)
        self._model_obj.callback = callback
        self._predictions = {}
        if test_split is None:
            test_split = set()
        else:
            test_split = set(test_split)
        keys = set(dataset.keys())
        train_split = list(keys.difference(test_split))
        test_split = list(test_split)

        X = np.array([dataset[k][0].flatten() for k in train_split])
        y = np.array([dataset[k][1] for k in train_split])
        validity = y >= 0
        X_test = np.array([dataset[k][0].flatten() for k in test_split]).reshape(-1, X.shape[-1])
        y_test = np.array([dataset[k][1] for k in test_split])
        test_validity = y_test >= 0
        callback("Starting fit...")
        start = time.time()
        self._model_obj.fit(X[validity], y[validity])
        pred = self._model_obj.predict(np.concatenate([X, X_test], axis=0))
        for s, p in zip(train_split + test_split, pred):
            self._predictions[s] = int(p)

        pred_test = [self._predictions[s] for s, valid in zip(test_split, test_validity) if valid]
        self.last_metrics = {
            "f1_score": round(f1_score(y_test[test_validity], pred_test), 2),
            "precision": round(precision_score(y_test[test_validity], pred_test), 2),
            "recall": round(recall_score(y_test[test_validity], pred_test), 2)
        }
        callback(f"Test scores: {self.last_metrics}")
        callback(f"Finished. Time elapsed: {time.time() - start}s")
        self.save()
        return self.last_metrics

    def get_predictions(self, samples=None):
        if samples is None:
            return self._predictions
        return {s: self._predictions.get(s, -1) for s in samples}

    def get_classes(self):
        out = set(self._predictions.values())
        out.add(-1)
        return out

    def get_activations(self, X, layer=-1):
        try:
            return self._model_obj.get_activations(X, layer=layer)
        except Exception as e:
            raise RuntimeError(e)
