from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from core.embedding import EmbeddingStore

from functools import partial
from threading import Event
from PIL import Image, ImageTk
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt


def sort_with_ranks(seq, ranks, reverse=True, return_rank=True):
    assert len(seq) == len(ranks)
    cat = list(zip(seq, ranks))
    sorted_cat = sorted(cat, key=lambda x: x[1], reverse=reverse)
    if return_rank:
        return sorted_cat
    return [cat_el[0] for cat_el in sorted_cat]


class SquarePad:
    def __init__(self, size=None):
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        out = v2.functional.pad(image, padding, 0, 'constant')
        if self.size is not None:
            out = v2.functional.resize(out, [self.size] * 2)
        return out


class MinResize:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, image):
        w, h = image.size
        min_wh = min(w, h)
        factor = self.min_size / min_wh
        return v2.functional.resize(image, [int(h * factor), int(w * factor)])


class PrecomputeDataset(Dataset):
    def __init__(self, samples, train=True):
        self.train = train
        self.samples = samples

        self.image_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            SquarePad(512),
            v2.RandomRotation(degrees=[-30, 30]),
            v2.RandomChoice([SquarePad(224),
                             v2.RandomCrop(450)], p=[0.7, 0.3]),

            v2.RandomApply([
                v2.ColorJitter(brightness=0.2, saturation=0.2, hue=0.2, contrast=0.2)
            ],
                p=0.7),
            v2.RandomGrayscale(p=0.3),
            v2.RandomChoice([v2.Identity(), v2.GaussianBlur(5)],
                            p=[0.75, 0.25])
        ])

        self.test_transform = SquarePad()

    def __len__(self):
        return len(self.samples)

    def show_sample(self, idx=0, n_runs=16):
        side_length = int(np.sqrt(n_runs))
        img_size = 224
        out = np.zeros((img_size * side_length, img_size * side_length, 3), dtype=np.uint8)
        for i in range(side_length):
            for k in range(side_length):
                img = np.asarray(SquarePad(img_size)(self.__getitem__(idx)[0]))
                out[i * img_size:(i + 1) * img_size, k * img_size:(k + 1) * img_size] = img.astype(np.uint8)
        return out

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # image
        image = Image.open(sample).convert(mode="RGB")
        if self.train:
            transformed = self.image_transform(image)
        else:
            transformed = self.test_transform(image)
        return transformed, sample


def precompute_collate_fn(sample_list: list):
    return tuple(zip(*sample_list))


class TrainDataset(Dataset):
    def __init__(self, samples, emb_store: EmbeddingStore, labels=None, spatial=True, squeeze=False, augs=True):
        super().__init__()
        assert labels is None or len(samples) == len(labels)
        self.samples = samples
        self.labels = labels
        self.embedder = partial(emb_store.get_image_embedding, spatial=spatial, squeeze=squeeze, augs=augs)
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if sample in self._cache:
            emb_full = self._cache[sample]
        else:
            emb_full = self.embedder(sample)
        if psutil.Process().memory_info()[0] / 2 ** 30 < 10:  # 10GB max
            # todo: eats gpu memory when .to(device) is called
            self._cache[sample] = emb_full
        if self.labels is None:
            return emb_full[0].float()
        else:
            random_id = torch.randint(0, emb_full.shape[0], (1,)).item()
            return emb_full[random_id].float(), torch.tensor(self.labels[idx]).float()

    def empty_cache(self):
        self._cache = {}
        import gc
        gc.collect()
        torch.cuda.empty_cache()


class ThreadKiller:
    def __init__(self):
        self.threads = {}

    def is_set(self, thread_id):
        if thread_id in self.threads:
            return self.threads[thread_id].is_set()
        return False

    def set(self, thread_id):
        self.threads[thread_id].set()

    def reset(self, thread_id):
        self.threads[thread_id] = Event()


def timer(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        print(f"Call to {func.__qualname__} elapsed: {time.time() - start:.4f}")
        return out

    return wrapped


def plt_non_interactive(func):
    def wrapped(*args, **kwargs):
        old_backend = plt.get_backend()
        plt.switch_backend("Agg")
        out = func(*args, **kwargs)
        plt.switch_backend(old_backend)
        return out

    return wrapped


@plt_non_interactive
def tk_plot(*args, fig: plt.Figure = None, ax: plt.Axes = None, render=True, func_name="plot", **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    getattr(ax, func_name)(*args, **kwargs)
    if render:
        fig.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        image = Image.fromarray(np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3))
        image = image.resize((400, 400))
        return ImageTk.PhotoImage(image)
    return fig, ax


def print_callback(value, end="\n", mode=None):
    print(value, end=end)


# globals
thread_killer = ThreadKiller()
