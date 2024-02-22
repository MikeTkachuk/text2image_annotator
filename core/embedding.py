import json
import time
from pathlib import Path
from typing import Union, List, Dict
from threading import Thread

from PIL import Image
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-large-patch14"

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder(model_name=MODEL_NAME):
    clip_model: CLIPModel = CLIPModel.from_pretrained(model_name, device_map=device)
    clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)

    def _embedder_func(obj):
        if isinstance(obj, (list, tuple)):
            element_instance = obj[0]
        else:
            element_instance = obj
        if isinstance(element_instance, str):
            inputs = clip_processor(text=obj, return_tensors="pt").to(device)
            return clip_model.get_text_features(**inputs)
        else:
            inputs = clip_processor(images=obj, return_tensors="pt").to(device)
            return clip_model.get_image_features(**inputs)

    return _embedder_func


class EmbeddingStore:
    def __init__(self, store_path, data_folder_path, model_name=MODEL_NAME, store_name=None):
        if store_name is None:
            store_name = model_name
        self.store_name = store_name
        self.model_name = model_name
        self.store_path = Path(store_path) / self.store_name
        self.data_folder_path = data_folder_path
        if not self.store_path.exists():
            self.store_path.mkdir(parents=True)
            with open(self.store_path / "embedder_spec.txt", "w") as spec_file:
                spec_file.write(model_name)
        else:
            with open(self.store_path / "embedder_spec.txt") as spec_file:
                assert model_name == spec_file.read(), "Could not load emb store with different model"

        self._embedder = None

        # transform is unimplemented feature of fine-tuning an mlp on top of embeddings
        self.transform_path = self.store_path / "transform.pt"
        if self.transform_path.exists():
            self.transform = torch.load(self.transform_path)
        else:
            self.transform = None

        self.tag_embeddings_path = self.store_path / "tag_embeddings.pt"
        if self.tag_embeddings_path.exists():
            self.tag_embeddings = torch.load(self.tag_embeddings_path)
        else:
            self.tag_embeddings = {}

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.model_name)
        return self._embedder

    def idle(self):
        del self._embedder
        self._embedder = None

    @torch.no_grad()
    def get_image_embedding(self, abs_path, load_only=False):
        rel_path = Path(abs_path).relative_to(self.data_folder_path)
        abs_emb_path = self.store_path / rel_path.parent / (rel_path.stem + ".pt")
        if abs_emb_path.exists():
            embedding = torch.load(abs_emb_path)
        else:
            if load_only:
                return None
            try:
                embedding = self.embedder(Image.open(abs_path))
                abs_emb_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(embedding, abs_emb_path)
            except:
                embedding = None
        return embedding

    @torch.no_grad()
    def add_tag(self, tag: Union[List[str], str]):
        if isinstance(tag, str):
            tag = [tag]
        new_embeddings = self.embedder(tag)
        for i, t in enumerate(tag):
            self.tag_embeddings[t] = new_embeddings[i]
        torch.save(self.tag_embeddings, self.tag_embeddings_path)

    def get_tag_embedding(self, tag: str):
        if tag not in self.tag_embeddings:
            self.add_tag(tag)

        return self.tag_embeddings[tag]

    @torch.no_grad()
    def get_tag_ranks(self, tag_list, image_abs_path):
        if not tag_list:
            return []
        image_embedding = self.get_image_embedding(image_abs_path)
        unprocessed_tags = [t for t in tag_list if t not in self.tag_embeddings]
        if unprocessed_tags:
            self.add_tag(unprocessed_tags)
        tag_embedding = torch.cat([self.tag_embeddings[t].reshape(1, -1) for t in tag_list], dim=0)
        cosines = image_embedding @ tag_embedding.T / (image_embedding.norm() * tag_embedding.norm(dim=1))

        return cosines.flatten().cpu().numpy().tolist()

    def precompute(self, samples, callback=None):
        for i, sample in enumerate(samples):
            self.get_image_embedding(sample)
            if callback is not None:
                callback(i)


class EmbeddingStoreRegistry:
    def __init__(self, save_dir, data_folder_path):
        self.save_dir = Path(save_dir)
        self.data_folder_path = data_folder_path

        self.stores: Dict[str, EmbeddingStore] = {}
        self._current_store: str = None
        self.load_state()

    @property
    def is_initialized(self):
        return self._current_store is not None and len(self.stores)

    def save_state(self):
        out = {store_name: store.model_name for store_name, store in self.stores.items()}
        with open(self.save_dir / "embstore_registry.json", "w") as file:
            json.dump(out, file)

    def load_state(self):
        if (self.save_dir / "embstore_registry.json").exists():
            with open(self.save_dir / "embstore_registry.json") as file:
                data = json.load(file)
            for store_name in data:
                self.stores[store_name] = EmbeddingStore(self.save_dir / ".emb_store",
                                                         self.data_folder_path,
                                                         model_name=data[store_name],
                                                         store_name=store_name
                                                         )

    def get_current_store(self):
        if self.is_initialized:
            return self.stores[self._current_store]
        return None

    def get_current_store_name(self):
        return self._current_store

    def add_store(self, model_path, store_name=None):
        if store_name is None:
            store_name = str(model_path)
        if store_name in self.stores:
            return False
        self.stores[store_name] = EmbeddingStore(self.save_dir / ".emb_store",
                                                 self.data_folder_path,
                                                 model_name=model_path,
                                                 store_name=store_name)
        self.save_state()
        self.choose_store(store_name)
        return store_name

    def choose_store(self, store_name):
        assert store_name in self.stores
        if self._current_store:
            self.get_current_store().idle()
        self._current_store = store_name
        print("store: ", self._current_store)
