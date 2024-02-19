from typing import List
from dataclasses import dataclass

from PIL import ImageTk, Image
import numpy as np
from sklearn.decomposition import PCA
import cv2 as cv

from core.embedding import EmbeddingStoreRegistry
from core.task import TaskRegistry
from config import CLUSTERING_IMG_SIZE

try:
    from tsnecuda import TSNE
    TSNE_CUDA_AVAILABLE = True
except (ImportError, Exception):
    TSNE_CUDA_AVAILABLE = False


@dataclass
class ClusteringResult:
    filenames: List[str]
    vectors: np.ndarray
    labels: List[int]


class Clustering:
    def __init__(self, embstore_registry: EmbeddingStoreRegistry, task_registry: TaskRegistry):
        self.embstore_registry = embstore_registry
        self.task_registry = task_registry
        self.samples = self.task_registry.get_samples()

        self._last_result: ClusteringResult = None

    def get_available_embeddings(self):
        embs = []
        available_samples = []
        labels = []
        for sample, label in zip(self.samples, self.task_registry.get_current_task().labels):
            emb = self.embstore_registry.get_current_store().get_image_embedding(sample, load_only=True)
            if emb is not None:
                embs.append(emb.cpu().numpy())
                available_samples.append(sample)
                labels.append(label)
        embs = np.concatenate(embs, axis=0)
        return available_samples, embs, labels

    def cluster(self, pca_components=50, random_state=42, tsne=True):
        samples, embs, labels = self.get_available_embeddings()
        pca_reduced = PCA(n_components=pca_components, random_state=random_state).fit_transform(embs)
        if TSNE_CUDA_AVAILABLE and tsne:
            out = TSNE(random_seed=random_state).fit_transform(pca_reduced)
        else:
            out = PCA(n_components=2, random_state=random_state).fit_transform(pca_reduced)
        out = out - out.min(axis=0)
        out = out / out.max(axis=0) * 0.8 + 0.1
        res = ClusteringResult(filenames=samples, vectors=out, labels=labels)
        self._last_result = res
        return res

    def draw_cluster_result(self, result=None):
        if result is None:
            result = self._last_result

        img = np.ones((CLUSTERING_IMG_SIZE, CLUSTERING_IMG_SIZE), dtype=np.uint8) * 255
        for vec in result.vectors:
            vec = vec * CLUSTERING_IMG_SIZE
            img[int(vec[1]), int(vec[0])] = 0
        img = cv.erode(img, kernel=np.ones((5,5), dtype=np.uint8))
        return ImageTk.PhotoImage(Image.fromarray(img))


def cluster(embstore, tasks):
    c = Clustering(embstore, tasks)
    c.cluster()
    return c.draw_cluster_result()