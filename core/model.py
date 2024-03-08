from pathlib import Path
import pickle
import inspect
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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
                                   "framework": "sklearn"}
        }

    @classmethod
    def from_template(cls, template_name, embstore_name, save_path):
        template = cls.get_templates()[template_name]
        return Model(template["class"], template.get("params", {}), template["framework"], embstore_name, save_path)

    def save(self):
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
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
            model_obj = pickle.load(file)
            model = Model(type(model_obj),
                          data["params"],
                          data["framework"],
                          data["embstore_name"],
                          data["path"])
            model._model_obj = model_obj
            model._predictions = data["predictions"]
        return model

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
        X_test = np.array([dataset[k][0].flatten() for k in test_split]).reshape(-1,X.shape[-1])
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
