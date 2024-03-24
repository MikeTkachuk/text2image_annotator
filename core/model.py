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


# todo: add fold it was trained with
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
        self._suggestions = []
        self._suggestion_cursor = -1

    @staticmethod
    def get_templates():
        return {
            "DecisionTreeClassifier": {"class": DecisionTreeClassifier,
                                       "framework": "sklearn"},
            "RandomForestClassifier": {"class": RandomForestClassifier,
                                       "framework": "sklearn"},
            "LogisticRegression": {"class": LogisticRegression,
                                   "framework": "sklearn"},
            "MLP": {"class": MLP, "framework": "torch"},
            "PoolConv": {"class": MLP,
                         "params": {"use_spatial": True, "batch_size": 64, },
                         "framework": "torch"}
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
            "suggestions": self._suggestions,
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
        model._suggestions = data["suggestions"]
        model.last_metrics = data["last_metrics"]
        return model

    def _k_fold(self, all_samples, test_samples=None, k=4):
        """

        :param all_samples:
        :param test_samples: if provided, considered as one of the folds and will be yielded last
        :param k: number of folds of size either len(test_samples) or len(all_samples_)//k
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
            ):
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
            pred_test = self._model_obj.predict(X_test)
            metrics.append({
                "f1_score": f1_score(y_test, pred_test),
                "precision": precision_score(y_test, pred_test),
                "recall": recall_score(y_test, pred_test)
            })
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
            proba, importance_scores = self._model_obj.get_importance(available_features)
            preds = self._model_obj.predict(None, probas=proba)
            unlabeled = [(smp, imp) for smp, imp in zip(available_keys, importance_scores) if dataset[smp] < 0]
            self._suggestions = list(zip(*sorted(unlabeled, key=lambda x: x[1], reverse=True)))[0]
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
        if samples is None:
            return self._predictions
        return [self._predictions.get(s, -1) for s in samples]

    def get_classes(self):
        out = set(self._predictions.values())
        out.add(-1)
        return out

    def get_activations(self, X, layer=-1):
        try:
            return self._model_obj.get_activations(X, layer=layer)
        except Exception as e:
            raise RuntimeError(e)

    def next_annotation_suggestion(self):
        self._suggestion_cursor = self._suggestion_cursor + 1
        if self._suggestion_cursor == len(self._suggestions):
            return None
        return self._suggestions[self._suggestion_cursor]

    def prev_annotation_suggestion(self):
        self._suggestion_cursor = self._suggestion_cursor - 1
        if self._suggestion_cursor < 0:
            return None
        return self._suggestions[self._suggestion_cursor]
