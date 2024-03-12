from pathlib import Path
import pickle
import inspect
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import torch
from torch.utils.data import DataLoader, Dataset


class MLP:
    def __init__(self, hidden_layers=(24, 5),
                 norm="batch",
                 activation="mish",
                 l2_weight=0.0,
                 learning_rate=0.01,
                 epochs=1000,
                 batch_size=-1,
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
        assert len(hidden_layers), "Provide at least one hidden layer"
        self.hidden_layers = hidden_layers
        self.norm = norm
        self.activation = activation
        self.l2_weight = l2_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self._categories = None
        self.model: torch.nn.ModuleList = None

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
        self.model = torch.nn.ModuleList()
        self.model.append(torch.nn.LazyLinear(self.hidden_layers[0]))
        for layer_dim in self.hidden_layers[1:]:
            self.model.append(_norm_lookup[self.norm]())
            self.model.append(_activations_lookup[self.activation]())
            self.model.append(torch.nn.LazyLinear(layer_dim))
        n_classes = len(self._categories)
        if n_classes < 3:
            n_classes -= 1
        assert n_classes > 0, f"Wrong class number, received {n_classes}"
        self.model.append(torch.nn.LazyLinear(n_classes, bias=False))

    def save(self, save_path):
        pt_path = Path(save_path)
        pt_path = pt_path.parent / (pt_path.stem + ".pt")
        torch.save(self.model, pt_path)
        model = self.model
        self.model = None
        with open(save_path, "wb") as file:
            pickle.dump(self, file)
        self.model = model

    @classmethod
    def load(cls, file_path):
        pt_path = Path(file_path)
        pt_path = pt_path.parent / (pt_path.stem + ".pt")
        model = None
        if pt_path.exists():
            model = torch.load(pt_path)
        with open(file_path, "rb") as file:
            obj = pickle.load(file)

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
                    return self.data[idx], self.labels[idx]
                else:
                    return self.data[idx]

        return _Dataset

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
            for batch in dataloader:
                data, labels = batch
                preds = self.model(data.to(self.device))
                loss = loss_func(preds, labels.to(self.device))
                loss.backward()
                optimizer.step()

    def predict_proba(self, X):
        dataset = self._dataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        all_logits = []
        for batch in dataloader:
            data, labels = batch
            preds = self.model(data.to(self.device))
            all_logits.append(preds)
        all_logits = torch.cat(all_logits, dim=0)
        if len(all_logits.shape) == 0 or all_logits.shape[-1] == 1:
            return torch.sigmoid(all_logits).flatten()
        else:
            assert len(all_logits.shape) == 2
            return torch.softmax(all_logits, dim=-1)

    def predict(self, X):
        probas = self.predict_proba(X)
        if len(probas.shape) > 1:
            return torch.argmax(probas, dim=-1)
        else:
            return torch.where(probas > 0.5, 1, 0)

    def get_activations(self, X, layer=-1):
        dataset = self._dataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        all_activations = []
        for batch in dataloader:
            data, labels = batch
            activations = [self.model[0](data.to(self.device))]
            for i in range(1, len(self.model)):
                x = self.model[i](x)
                if isinstance(self.model[i], (torch.nn.Linear, torch.nn.LazyLinear)):
                    activations.append(x)
            all_activations.append(activations[layer])
        return torch.cat(all_activations, dim=0)


# todo: add mlp, add option to do clustering on mlp
class Model:
    def __init__(self, model, params, framework, embstore_name, save_path):
        self.model = model
        self.params = params
        self.framework = framework
        self.save_path = Path(save_path)
        self.embstore_name = embstore_name

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

    def save(self):
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        if self.framework == "torch":
            self._model_obj.save(self.save_path)
        else:
            with open(self.save_path, "wb") as file:
                pickle.dump(self._model_obj, file)

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

    def __dict__(self):
        self.save()
        return {"path": str(self.save_path),
                "params": self.params,
                "framework": self.framework,
                "embstore_name": self.embstore_name,
                "predictions": self._predictions
                }

    @classmethod
    def load(cls, data):
        with open(data["path"], "rb") as file:
            if data["framework"] == "torch":
                model_obj = MLP.load(file)
            else:
                model_obj = pickle.load(file)
            model = Model(type(model_obj),
                          data["params"],
                          data["framework"],
                          data["embstore_name"],
                          data["path"])
            model._model_obj = model_obj
            model._predictions = data["predictions"]
        return model

    # todo:
    #  - fit existing samples
    #  - compute validation metrics
    #  - show model emb clusters
    #  - show decision boundary
    #  - suggest based on uncertainty, cluster coverage
    def fit(self, dataset: Dict[str, Tuple[Any, int]], train_split=None):
        print(self.params)
        self._model_obj = self.model(**self.params)
        self._predictions = {}
        if train_split is None:
            train_split = set(dataset.keys())
        keys = set(dataset.keys())
        test_split = list(keys.difference(train_split))
        train_split = list(train_split)

        X = np.array([dataset[k][0].flatten() for k in train_split])
        y = np.array([dataset[k][1] for k in train_split])
        validity = y >= 0
        X_test = np.array([dataset[k][0].flatten() for k in test_split]).reshape(-1, X.shape[-1])
        y_test = np.array([dataset[k][1] for k in test_split])
        self._model_obj.fit(X[validity], y[validity])
        pred = self._model_obj.predict(np.concatenate([X, X_test], axis=0))
        for s, p in zip(train_split, pred):
            self._predictions[s] = int(p)

        return {}

    def get_predictions(self, samples=None):
        if samples is None:
            return self._predictions
        return {s: self._predictions.get(s, -1) for s in samples}

    def get_classes(self):
        out = set(self._predictions.values())
        out.add(-1)
        return out
