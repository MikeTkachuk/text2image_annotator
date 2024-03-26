from typing import List, Dict, Union
from dataclasses import dataclass

import tqdm
from PIL import ImageTk, Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import cv2 as cv
from matplotlib.pyplot import get_cmap

from core.embedding import EmbeddingStoreRegistry
from core.task import TaskRegistry
from config import CLUSTERING_IMG_SIZE
from core.utils import TrainDataset

try:
    from tsnecuda import TSNE

    TSNE_CUDA_AVAILABLE = True
except (ImportError, Exception):
    import warnings
    warnings.warn("Tsne-cuda unavailable, skipping import.")
    TSNE_CUDA_AVAILABLE = False


@dataclass
class ClusteringResult:
    filenames: List[str]
    filename_to_id: Dict[str, int]
    vectors: np.ndarray
    labels: List[int]
    neighbor_tree: NearestNeighbors
    base_plot: np.ndarray
    neighbors_plot: np.ndarray = None
    predictions: List[int] = None
    split: List[str] = None  # train/val
    version: List[str] = None  # orig/aug_0/...


def draw_alpha_rect(img, pts, color=(50, 255, 100), alpha=0.5):
    overlay = np.zeros_like(img)
    overlay = cv.fillConvexPoly(overlay, cv.convexHull(pts), color)
    overlay = cv.dilate(overlay, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    return cv.addWeighted(img, 1 - alpha, overlay, alpha, 0)


class Clustering:
    def __init__(self, embstore_registry: EmbeddingStoreRegistry, task_registry: TaskRegistry):
        self.embstore_registry = embstore_registry
        self.task_registry = task_registry

        self._last_result: ClusteringResult = None

    def get_available_embeddings(self, use_model_features=False, layer=-2, augs=True):
        embs = []
        available_samples = []
        labels = []
        versions = []
        self.task_registry.update_current_task()
        store = self.embstore_registry.get_current_store()
        model = self.task_registry.get_current_model()
        use_model_features = use_model_features and model is not None
        use_spatial = use_model_features and model.params.get("use_spatial", False)
        if use_model_features:
            available_samples, labels = list(zip(*[
                (s, l) for s, l in self.task_registry.get_current_labels().items() if store.embedding_exists(s)
            ]))
            dataset = TrainDataset(available_samples,
                                   store,
                                   spatial=use_spatial,
                                   augs=False)  # augs are not used by design
            embs = model.get_activations(dataset, layer=layer)
            versions = ["orig"] * len(embs)
        else:
            for sample, label in tqdm.tqdm(self.task_registry.get_current_labels().items(),
                                           desc="Loading embeddings"):
                emb = store.get_image_embedding(sample, load_only=True, spatial=use_spatial, augs=augs)
                if emb is not None:
                    embs.append(emb.cpu().numpy())
                    available_samples.extend([sample] * emb.shape[0])
                    labels.extend([label] * emb.shape[0])
                    version = [f"aug_{i - 1}" for i in range(emb.shape[0])]
                    version[0] = "orig"
                    versions.extend(version)
            embs = np.concatenate(embs, axis=0)
        predictions = model.get_predictions(available_samples) if (model is not None and use_model_features) else None
        val_samples = set(self.task_registry.get_current_task().validation_samples)
        splits = ["val" if s in val_samples else "train" for s in available_samples]
        return available_samples, embs, labels, predictions, splits, versions

    def update_labels(self, use_predictions=False):
        if self._last_result is None:
            return
        model = self.task_registry.get_current_model()
        if use_predictions and model is not None:
            label_lookup = model.get_predictions()
            self._last_result.predictions = [label_lookup.get(s, -1) for s in self._last_result.filenames]
            self._last_result.base_plot = self.draw_cluster_result(self._last_result.vectors,
                                                                   self._last_result.predictions)
        else:
            label_lookup = self.task_registry.get_current_labels()
            self._last_result.labels = [label_lookup.get(s, -1) for s in self._last_result.filenames]
            self._last_result.base_plot = self.draw_cluster_result(self._last_result.vectors,
                                                                   self._last_result.labels)

    def cluster(self,
                pca_components=50,
                random_state=42,
                tsne=True,
                use_model_features=True,
                layer=-2,
                augs=False
                ):
        if not self.task_registry.is_initialized:
            return

        # free cuda memory
        self.embstore_registry.get_current_store().idle()
        if not self.embstore_registry.is_initialized:
            return
        samples, embs, labels, predictions, splits, versions = self.get_available_embeddings(use_model_features,
                                                                                             layer=layer,
                                                                                             augs=augs)
        if pca_components < embs.shape[-1]:
            pca_reduced = PCA(n_components=pca_components, random_state=random_state, svd_solver="full").fit_transform(
                embs)
        else:
            pca_reduced = embs

        if TSNE_CUDA_AVAILABLE and tsne:
            out = TSNE(random_seed=random_state).fit_transform(pca_reduced)
        else:
            out = PCA(n_components=2, random_state=random_state).fit_transform(pca_reduced)
        out = out - out.min(axis=0)
        out = out / out.max(axis=0) * 0.8 + 0.1
        n_tree = NearestNeighbors().fit(out)
        base_plot = self.draw_cluster_result(out, labels)
        res = ClusteringResult(filenames=samples,
                               filename_to_id=dict(zip(samples, range(len(samples)))),
                               vectors=out,
                               labels=labels,
                               predictions=predictions,
                               split=splits,
                               version=versions,
                               neighbor_tree=n_tree,
                               base_plot=base_plot
                               )
        self._last_result = res
        return res

    def get_label_color(self, value: Union[str, int]):
        if value == "selection":
            return (255, 100, 50)

        if value == -1:
            return (100, 100, 100)

        if value == -5:
            return (100, 25, 25)
        task = self.task_registry.get_current_task()
        color = get_cmap("gist_rainbow")(value / len(task.categories_full))[:3]
        color = np.array(color) * 255
        return tuple(color.astype(np.uint8).tolist())

    def draw_cluster_result(self, vectors, labels):
        img = np.ones((CLUSTERING_IMG_SIZE, CLUSTERING_IMG_SIZE, 3), dtype=np.uint8) * 255
        zipped = sorted(zip(vectors, labels), key=lambda x: x[1])
        for vec, label in zipped:
            vec = vec * CLUSTERING_IMG_SIZE
            img = cv.circle(img, [int(vec[0]), int(vec[1])],
                            1, self.get_label_color(label)[:3], -1, cv.LINE_AA)
        return img

    def get_base_plot(self):
        if self._last_result is None:
            img = np.ones((CLUSTERING_IMG_SIZE, CLUSTERING_IMG_SIZE), dtype=np.uint8) * 255
            return ImageTk.PhotoImage(Image.fromarray(img))
        return ImageTk.PhotoImage(Image.fromarray(self._last_result.base_plot))

    def get_legend(self, img_size=32):
        out = []
        if self._last_result is None:
            return out
        for label in sorted(set(self._last_result.labels)):
            sample = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            sample[:, :] = np.array(self.get_label_color(label))
            out.append((sample, self.task_registry.get_current_task().label_name(label)))
        return out

    def get_nearest_neighbors(self, x, y, n_neighbors):
        if self._last_result is None:
            return None

        x /= CLUSTERING_IMG_SIZE
        y /= CLUSTERING_IMG_SIZE
        if n_neighbors < 1:
            res = self._last_result.neighbor_tree.radius_neighbors([[x, y]],
                                                                   radius=n_neighbors, sort_results=False)
        else:
            res = self._last_result.neighbor_tree.kneighbors([[x, y]], n_neighbors=int(n_neighbors))

        res = (res[0][0], res[1][0])
        res = sorted(zip(*res), key=lambda x: x[0])
        res_vectors = [(CLUSTERING_IMG_SIZE * self._last_result.vectors[i[1]]).astype(np.int32) for i in res]
        if res_vectors:
            viz = draw_alpha_rect(self._last_result.base_plot, np.array([res_vectors]), alpha=0.25)
        else:
            viz = self._last_result.base_plot
        self._last_result.neighbors_plot = viz

        task = self.task_registry.get_current_task()
        ids = list(zip(*res))[1] if res else []
        return (ids,
                [self._last_result.filenames[i] for i in ids],
                [task.label_name(self._last_result.labels[i]) for i in ids],
                [task.label_name(self._last_result.predictions[i]) if self._last_result.predictions else None for i in ids],
                [self._last_result.split[i] for i in ids],
                [self._last_result.version[i] for i in ids],
                ImageTk.PhotoImage(Image.fromarray(viz)))

    def draw_selection(self, ids):
        img = np.copy(self._last_result.neighbors_plot)
        for i in ids:
            vec = self._last_result.vectors[i]
            vec = vec * CLUSTERING_IMG_SIZE
            img = cv.circle(img, [int(vec[0]), int(vec[1])],
                            2, self.get_label_color("selection")[:3], -1, cv.LINE_AA)
        return ImageTk.PhotoImage(Image.fromarray(img))

    def get_data_of_sample(self, filename: str):
        if self._last_result is None:
            return None
        sample_id = self._last_result.filename_to_id[filename]
        return self._last_result.vectors[sample_id]
