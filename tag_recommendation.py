from pathlib import Path
from typing import Union, List

from PIL import Image
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


def optimize_metric_func():
    pass


class EmbeddingStore:
    def __init__(self, store_path, model_name=MODEL_NAME):
        store_path = Path(store_path)
        self.store_path = store_path
        if not self.store_path.exists():
            self.store_path.mkdir(parents=True)
            with open(self.store_path / "embedder_spec.txt", "w") as spec_file:
                spec_file.write(model_name)
        else:
            with open(self.store_path / "embedder_spec.txt") as spec_file:
                assert model_name == spec_file.read(), "Could not load emb store with different model"

        self.embedder = get_embedder(model_name)

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

    @torch.no_grad()
    def get_image_embedding(self, abs_path, folder_path):
        rel_path = Path(abs_path).relative_to(folder_path)
        abs_emb_path = self.store_path / rel_path.parent / (rel_path.stem + ".pt")
        if abs_emb_path.exists():
            embedding = torch.load(abs_emb_path)
        else:
            embedding = self.embedder(Image.open(abs_path))
            abs_emb_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embedding, abs_emb_path)
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
    def get_tag_ranks(self, tag_list, image_abs_path, folder_path):
        image_embedding = self.get_image_embedding(image_abs_path, folder_path)
        unprocessed_tags = [t for t in tag_list if t not in self.tag_embeddings]
        if unprocessed_tags:
            self.add_tag(unprocessed_tags)
        tag_embedding = torch.cat([self.tag_embeddings[t].reshape(1, -1) for t in tag_list], dim=0)
        cosines = image_embedding @ tag_embedding.T / (image_embedding.norm() * tag_embedding.norm(dim=1))

        return cosines.squeeze().cpu().numpy().tolist()

    # TODO connect session config and emb store to do finetuning