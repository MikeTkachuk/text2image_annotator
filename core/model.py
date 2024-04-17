from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.embedding import EmbeddingStore

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

from core.mlp import MLP
from core.utils import TrainDataset


class Model:
    """Predictor wrapper to store model generator parameters, base class, last metrics,
    predictions, and annotation suggestions if available"""
    def __init__(self, model, params, framework, embstore_name, save_path):
        self.model = model
        self.params = params
        self.framework = framework
        self.save_path = Path(save_path)
        self.embstore_name = embstore_name

        self.last_metrics = {}
        self._model_obj = self.model(**self.params)
        self._predictions: Dict[str, int] = {}
        self._metadata: Dict[str, dict] = {}
        self._suggestions = []
        self._suggestion_cursor = -1

    @staticmethod
    def get_default_templates():
        """Returns default model templates. Can be customized"""
        return {
            "DecisionTreeClassifier": {"class": "DecisionTreeClassifier",
                                       "framework": "sklearn"},
            "RandomForestClassifier": {"class": "RandomForestClassifier",
                                       "framework": "sklearn"},
            "LogisticRegression": {"class": "LogisticRegression",
                                   "framework": "sklearn"},
            "MLP": {"class": "MLP", "framework": "torch"},
            "PoolConv": {"class": "MLP",
                         "params": {"use_spatial": True},
                         "framework": "torch"}
        }

    @staticmethod
    def template_class_map():
        return dict((c.__name__, c) for c in
                    [MLP, DecisionTreeClassifier,
                     RandomForestClassifier, LogisticRegression])

    @classmethod
    def from_template(cls, template: dict, embstore_name: str, save_path):
        """Initialize class and params from template"""
        class_map = cls.template_class_map()
        return Model(class_map[template["class"]], template.get("params", {}),
                     template["framework"], embstore_name, save_path)

    def to_template(self):
        """Extract template info"""
        return {"class": self.model.__name__,
                "params": self.params,
                "framework": self.framework}

    def __dict__(self):
        out = {
            "params": self.params,
            "framework": self.framework,
            "embstore_name": self.embstore_name,
            "predictions": self._predictions,
            "last_metrics": self.last_metrics,
        }
        return out

    def save(self):
        """Serializes model metadata and the model itself"""
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
            "metadata": self._metadata,
            "suggestions": self._suggestions,
            "last_metrics": self.last_metrics
        }
        with open(config_path, "w") as file:
            json.dump(config, file)

    def get_model_signature(self):
        """Parses predictor init function and returns a dict of kwargs and their default values"""
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
        model._predictions = data.get("predictions", {})
        model._metadata = data.get("metadata", {})
        model._suggestions = data.get("suggestions", [])
        model.last_metrics = data.get("last_metrics", {})
        return model

    @staticmethod
    def _k_fold(all_samples, test_samples=None, k=4):
        """
        Fold generator. Shuffles the sample pool. If provided, test samples are separated from
         the pool and appended to the shuffled result. After that yields evenly spaced folds, can be overlapped.
        :param all_samples:
        :param test_samples: if provided, considered as one of the folds and will be yielded last
        :param k: number of folds of size either len(test_samples) if provided or len(all_samples_)//k
        :return:
        """
        shuffle_index = np.random.permutation(len(all_samples))
        all_samples = [all_samples[i] for i in shuffle_index]
        if k == 1 and test_samples is None:
            yield all_samples, []

        if test_samples is not None:
            fold_size = len(test_samples)
            test_set = set(test_samples)
            all_samples = [s for s in all_samples if s not in test_set] + test_samples
        else:
            fold_size = len(all_samples) // k

        for i in range(k):
            if i == k - 1 and test_samples is not None:
                yield all_samples[:len(all_samples) - len(test_samples)], test_samples
            else:
                step_size = (len(all_samples) - fold_size) // (k - 1)  # moved inside to avoid zero division
                yield (all_samples[:i * step_size] + all_samples[fold_size + i * step_size:],
                       all_samples[i * step_size:fold_size + i * step_size])

    # todo: (goes by active learning)
    #  - fit existing samples
    #  - compute validation metrics
    #  - show model emb clusters
    #  - show decision boundary
    #  - suggest based on uncertainty, cluster coverage, class balance
    #  - suggestions should be a stream of images sorted by impact/relevance
    # todo: measure suggestion efficiency by validating on existing dataset
    def fit(self, dataset: Dict[str, int], emb_store: EmbeddingStore,
            test_split=None,
            callback=None,
            kfold=None,
            use_augs=True
            ) -> dict:
        """
        Currently supports torch and sklearn training routes. Fits predictors,
         evaluates on test data if provided, and populates predictions, suggestions and other metadata.
         Also resets suggestion cursor and saves self on disk.

        :param dataset: sample paths and their labels
        :param emb_store:
        :param test_split: iterable of sample paths to be used for final evaluation
        :param callback: print-like func str -> None for logging purposes. default is print
        :param kfold: number of folds, default is 1
        :param use_augs: bool, augmented data is used if True, passed into emb_store
        :return: final metrics
        """
        if callback is None:
            callback = print
        if kfold is None:
            kfold = 1

        callback(f"Model params: {self.params}")
        self._model_obj = self.model(**self.params)
        if self.framework == "torch":
            self._model_obj.callback = callback
        self._predictions = {}
        use_augs = self.params.get("augs", use_augs)
        labeled_data = [s for s in dataset if dataset[s] >= 0]

        callback(f"Framework route: {self.framework}")
        start = time.time()
        metrics = []
        for i, (train_split, test_split) in enumerate(self._k_fold(labeled_data, test_samples=test_split, k=kfold)):
            callback(f"Fold #{i}")
            if self.framework == "torch":
                X = TrainDataset(train_split, emb_store, labels=[dataset[s] for s in train_split],
                                 spatial=self.params.get("use_spatial", False), augs=use_augs)
                y = [dataset[s] for s in train_split]
                X_test = TrainDataset(test_split, emb_store,
                                      spatial=self.params.get("use_spatial", False), augs=False)
                y_test = [dataset[s] for s in test_split]

                def test_func():
                    if not len(X_test):
                        return
                    _pred_test = self._model_obj.predict(X_test)
                    m = {
                        "f1_score": f1_score(y_test, _pred_test),
                        "precision": precision_score(y_test, _pred_test),
                        "recall": recall_score(y_test, _pred_test)
                    }
                    callback(f'epoch eval metrics: {m}')

                callback("Starting fit...")
                self._model_obj.fit(X, y, test_func=test_func)
            else:
                callback(f"Loading embeddings...")
                X, y = [], []
                for sample in train_split:
                    X.append(emb_store.get_image_embedding(sample, load_only=True,
                                                           augs=use_augs).cpu().numpy())
                    y.extend([dataset[sample]] * X[-1].shape[0])
                X = np.concatenate(X, axis=0)
                y = np.array(y)
                X_test, y_test = [], []
                for sample in test_split:
                    X_test.append(emb_store.get_image_embedding(sample, load_only=True,
                                                                augs=False).cpu().numpy())
                    y_test.extend([dataset[sample]] * X_test[-1].shape[0])
                X_test = np.concatenate(X_test, axis=0)
                y_test = np.array(y_test)
                callback("Starting fit...")
                self._model_obj.fit(X, y)
            if len(X_test):
                pred_test = self._model_obj.predict(X_test)
                metrics.append({
                    "f1_score": f1_score(y_test, pred_test),
                    "precision": precision_score(y_test, pred_test),
                    "recall": recall_score(y_test, pred_test)
                })
            else:
                metrics.append({})
            callback(f"Fold scores: {metrics[-1]}")
        self.last_metrics = {k: round(sum([m[k] for m in metrics]) / len(metrics), 2) for k in metrics[0]}
        callback(f"Total test scores: {self.last_metrics}")
        callback(f"Predicting available data...")
        available_keys = list(dataset.keys())
        if self.framework == "torch":
            available_features = TrainDataset(available_keys, emb_store, spatial=self.params.get("use_spatial", False),
                                              augs=False)
        else:
            available_features = np.concatenate([emb_store.get_image_embedding(k, load_only=True,
                                                                               augs=False).cpu().numpy() for k in
                                                 available_keys])

        if hasattr(self._model_obj, "get_importance"):
            proba, meta, order = self._model_obj.get_importance(available_features)
            self._metadata = dict(zip(available_keys, meta))
            preds = self._model_obj.predict(None, probas=proba)
            unlabeled = [available_keys[i] for i in order if dataset[available_keys[i]] < 0]
            self._suggestions = unlabeled
        else:
            preds = self._model_obj.predict(available_features)
            self._suggestions = []
        for k, p in zip(available_keys, preds):
            self._predictions[k] = int(p)

        callback(f"Finished. Time elapsed: {time.time() - start}s")
        self._suggestion_cursor = -1
        self.save()
        return self.last_metrics

    def get_predictions(self, samples=None):
        """Returns a map from sample paths to respective predictions.
                Maps internally if samples iterable is provided"""
        if samples is None:
            return self._predictions
        return [self._predictions.get(s, -1) for s in samples]

    def get_metadata(self, samples=None):
        """Returns a map from sample paths to respective metadata.
                Maps internally if samples iterable is provided"""
        if samples is None:
            return self._metadata
        return [self._metadata.get(s) for s in samples]

    def get_classes(self):
        """Returns a set of classes used in current predictions. Not labeled class is included"""
        out = set(self._predictions.values())
        out.add(-1)  # for not labeled case
        return out

    def get_activations(self, X, layer=-1):
        """Returns intermediate activations if defined.
        :param X: data to be predicted
        :param layer: int, index of linear layer to collect from
        """
        try:
            return self._model_obj.get_activations(X, layer=layer)
        except Exception as e:
            raise RuntimeError(e)

    def next_annotation_suggestion(self):
        """Returns the next suggested sample to annotate"""
        self._suggestion_cursor = self._suggestion_cursor + 1
        if self._suggestion_cursor == len(self._suggestions):
            return None
        return self._suggestions[self._suggestion_cursor]

    def prev_annotation_suggestion(self):
        """Returns the previous suggested sample to annotate"""
        self._suggestion_cursor = self._suggestion_cursor - 1
        if self._suggestion_cursor < 0:
            return None
        return self._suggestions[self._suggestion_cursor]
