import json
from pathlib import Path
import shutil
from typing import Union, List, Dict

from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor, ViTMAEForPreTraining, AutoImageProcessor

from core.utils import PrecomputeDataset, precompute_collate_fn, TrainDataset, sort_with_ranks

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder(model_name):
    """Embedder factory function. Should be customized to support other architectures and APIs.
        Currently supports hugging face transformers API.
        An image embedder func should return a dictionary of optional features (see code below).
        A text embedder returns plain vector tensors.
    """
    if "clip" in model_name:
        clip_model: CLIPModel = CLIPModel.from_pretrained(model_name, device_map=device)
        clip_model.eval()
        clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)

        def _img_func(obj):
            inputs = clip_processor(images=obj, return_tensors="pt").to(device)
            hidden_states = clip_model.vision_model(inputs.pixel_values,
                                                    output_hidden_states=True)[0]
            n_patches = clip_model.config.vision_config.image_size // clip_model.config.vision_config.patch_size
            return {"pooled": hidden_states[:, 0, :, None, None],
                    "spatial": hidden_states[:, 1:, :].transpose(1, 2).unflatten(-1, (n_patches, n_patches))}

        def _text_func(obj):
            inputs = clip_processor(text=obj, return_tensors="pt").to(device)
            return clip_model.get_text_features(**inputs)

        return _img_func, _text_func
    else:  # ViTMAE for now
        img_model: ViTMAEForPreTraining = ViTMAEForPreTraining.from_pretrained(model_name, device_map=device)
        img_model.eval()
        img_model.config.mask_ratio = 0.0
        img_processor = AutoImageProcessor.from_pretrained(model_name)

        def _img_func(obj):
            inputs = img_processor(images=obj, return_tensors="pt").to(device)
            hidden_states = img_model.vit(**inputs).last_hidden_state
            n_patches = img_model.config.image_size // img_model.config.patch_size
            return {"pooled": hidden_states[:, 0, :, None, None],
                    "spatial": hidden_states[:, 1:, :].transpose(1, 2).unflatten(-1, (n_patches, n_patches))}

        return _img_func, None


class EmbeddingStore:
    """Entity to handle embedding computation and serialization"""
    def __init__(self, store_path, data_folder_path, model_name, store_name=None):
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

        self.tag_embeddings_path = self.store_path / "tag_embeddings.pt"
        if self.tag_embeddings_path.exists():
            self.tag_embeddings = torch.load(self.tag_embeddings_path)
        else:
            self.tag_embeddings = {}

    @property
    def embedder(self):
        """Singleton embedder instance"""
        if self._embedder is None:
            self._embedder = get_embedder(self.model_name)
        return self._embedder

    def idle(self):
        """Offloads embedder to free memory"""
        if self._embedder is not None:
            print("Embedder idle state requested")
            del self._embedder
            import gc
            gc.collect()
            self._embedder = None
            torch.cuda.empty_cache()

    def embedding_exists(self, abs_path):
        """Returns true if the data sample embedding was already computed.
         Assumes integrity and checks default pooled original non-augmented embedding

         :param abs_path: path-like, path to image sample
         """
        return self._emb_path(abs_path).exists()

    def _emb_path(self, sample_path, spatial=False, aug_id=None):
        """Linked to embedder return spec."""
        rel_path = Path(sample_path).relative_to(self.data_folder_path)
        emb_subfolder = self.store_path / rel_path.parent / rel_path.stem
        if spatial:
            path = emb_subfolder / "spatial"
        else:
            path = emb_subfolder / "pooled"

        if aug_id is not None:
            path = path / f"aug_{aug_id}.pt"
        else:
            path = path / "orig.pt"
        return path

    def _save_embedder_output(self, sample_path,
                              output: Dict[str, torch.Tensor],
                              aug_id=None, dtype=torch.float16):
        """Linked to embedder return spec."""

        pooled = output["pooled"]
        if len(pooled.shape) > 3:
            pooled = pooled[0]
        path = self._emb_path(sample_path, aug_id=aug_id)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(pooled.to(dtype), path)
        spatial = output["spatial"]
        if len(spatial.shape) > 3:
            spatial = spatial[0]
        path = self._emb_path(sample_path, spatial=True, aug_id=aug_id)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(spatial.to(dtype), path)

    @torch.no_grad()
    def get_image_embedding(self, abs_path,
                            load_only=False,
                            normalize=False,
                            spatial=False,
                            squeeze=True,
                            augs=False):
        """
        Image embedding getter. Computes non-augmented version of image embedding if not cached already.
        :param abs_path: path to image
        :param load_only: loads from cache, returns None if does not exist
        :param normalize: normalizes along embedding dimension (1)
        :param spatial: if True, returns features of shape [None, embedding_size, h, w],
                        else returns pooled features with h, w = 1
        :param squeeze: if spatial is False, new shape is [None, embedding_size]
        :param augs: int or bool. if bool(augs) is False, returns features for original image,
                    else returns [:augs] of augmented features or all of them if augs is True
        :return:
        """
        abs_emb_path = self._emb_path(abs_path, spatial=spatial)
        if not augs:
            augs = 0
        elif not isinstance(augs, bool):
            augs = int(augs)
        else:
            augs = -1

        if abs_emb_path.exists():
            aug_count = len(list(abs_emb_path.parent.glob("*.pt"))) - 1
            if augs >= 0:
                aug_count = min(augs, aug_count)
            aug_paths = [self._emb_path(abs_path, spatial=spatial, aug_id=i) for i in range(aug_count)]
            embedding = torch.stack([torch.load(abs_emb_path)] + [torch.load(p) for p in aug_paths])
        else:
            if load_only:
                return None
            try:
                output = self.embedder[0](Image.open(abs_path).convert("RGB"))
                self._save_embedder_output(abs_path, output)
                return self.get_image_embedding(abs_path, load_only=True, normalize=normalize,
                                                spatial=spatial, squeeze=squeeze, augs=augs)
            except Exception as e:
                print(e, abs_path)
                import traceback
                traceback.print_exception(e)
                return None

        if squeeze:
            embedding = embedding.flatten(1)

        if normalize and embedding is not None:
            embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
        return embedding.detach()

    @torch.no_grad()
    def add_tag(self, tag: Union[List[str], str]):
        """Computes text embedding and caches it."""
        if isinstance(tag, str):
            tag = [tag]
        if self.embedder[1] is None:
            raise RuntimeError("Text embedder not available")
        new_embeddings = self.embedder[1](tag)
        for i, t in enumerate(tag):
            self.tag_embeddings[t] = new_embeddings[i]
        torch.save(self.tag_embeddings, self.tag_embeddings_path)

    def get_tag_embedding(self, tag: str):
        """Text embedding getter."""
        if tag not in self.tag_embeddings:
            self.add_tag(tag)

        return self.tag_embeddings[tag]

    @torch.no_grad()
    def get_tag_ranks(self, tag_list, image_abs_path) -> List[float]:
        """Provided image path and text candidates, compute clip-like similarities"""
        if not tag_list:
            return []
        image_embedding = self.get_image_embedding(image_abs_path)
        unprocessed_tags = [t for t in tag_list if t not in self.tag_embeddings]
        if unprocessed_tags:
            self.add_tag(unprocessed_tags)
        tag_embedding = torch.cat([self.tag_embeddings[t].reshape(1, -1) for t in tag_list], dim=0)
        cosines = image_embedding @ tag_embedding.T / (image_embedding.norm() * tag_embedding.norm(dim=1))

        return cosines.flatten().cpu().numpy().tolist()

    @torch.no_grad()
    def precompute(self, samples, callback=None, batch_size=1, aug_per_img=0, append=True):
        """Precomputes embeddings for a chunk of image data in a batched way.
        See core.utils.PrecomputeDataset for augmentations info and customization
        :param samples: iterable of image paths
        :param callback: optional logging function. Will be called as callback(num_of_processed)
        :param batch_size: int,
        :param aug_per_img: int, number of embedding variations per image
        :param append: if False, unlinks all existing embeddings of the images to be processed and computes them anew
                        else appends the augmented embedding variations to the cache
        """
        raw_dataset = PrecomputeDataset(samples, train=False)
        raw_dataloader = DataLoader(raw_dataset,
                                    batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True,
                                    collate_fn=precompute_collate_fn,
                                    prefetch_factor=3)
        if aug_per_img > 0:
            aug_samples = sum([samples[i*batch_size:(i+1)*batch_size] * aug_per_img
                               for i in range(len(samples)//batch_size + 1)], [])
            aug_dataset = PrecomputeDataset(aug_samples, train=True)
            aug_dataloader = DataLoader(aug_dataset,
                                        batch_size=batch_size,
                                        num_workers=4,
                                        pin_memory=True,
                                        collate_fn=precompute_collate_fn,
                                        prefetch_factor=3)
            aug_dataloader_iterator = iter(aug_dataloader)
        else:
            aug_dataloader_iterator = None
        count = 0
        for i, batch in enumerate(raw_dataloader):
            try:
                images, paths = batch
                true_embedding = self.embedder[0](images)
                aug_results = []
                if aug_dataloader_iterator is not None:
                    for _ in range(aug_per_img):
                        aug_batch = next(aug_dataloader_iterator)
                        assert aug_batch[1] == paths
                        aug_results.append(self.embedder[0](aug_batch[0]))

                for sample_id in range(len(paths)):
                    pooled_subfolder = self._emb_path(paths[sample_id])
                    if not append:
                        if pooled_subfolder.parent.exists():
                            shutil.rmtree(pooled_subfolder.parent)
                        aug_start_id = 0
                    else:
                        aug_start_id = len(list(pooled_subfolder.glob("*.pt"))) - 1
                        aug_start_id = max(0, aug_start_id)

                    to_save = {k: v[sample_id] for k, v in true_embedding.items()}
                    self._save_embedder_output(paths[sample_id], to_save)
                    for aug_i, aug_res in enumerate(aug_results):
                        to_save = {k: aug_res[k][sample_id] for k in aug_res}
                        self._save_embedder_output(paths[sample_id], to_save, aug_id=aug_start_id + aug_i)

                count += len(paths)
            except Exception as e:
                import traceback
                traceback.print_exc()
            if callback is not None:
                callback(i * batch_size)

        return count

    @torch.no_grad()
    def get_duplicates(self, samples, batch_size=512, eps=1E-4):
        """Computes similarity matrix in chunks and returns pairs of samples with euclid distance fewer than eps

        :param samples: iterable of image paths
        :param batch_size: int,
        :param eps: non-negative float, distance threshold

        :returns list of path pairs sorted by difference (desc)
        """
        dataset = TrainDataset(samples, self, spatial=False, squeeze=True, augs=False)
        dataloader1 = DataLoader(dataset, batch_size=batch_size)
        dataloader2 = DataLoader(dataset, batch_size=batch_size)
        path_pairs = []
        scores = []
        for i, batch1 in enumerate(dataloader1):
            batch1 = batch1.to(device)
            for k, batch2 in enumerate(dataloader2):
                batch2 = batch2.to(device)
                diff = torch.norm(batch1.unsqueeze(1)-batch2.unsqueeze(0), dim=-1)
                mask = torch.ones_like(diff).triu().bool()
                diff = torch.where(mask, 64, diff)
                for row, col in torch.nonzero(diff < eps):
                    path_pairs.append(
                        (samples[i*batch_size + row], samples[k*batch_size + col])
                    )
                    scores.append(diff[row, col].item())
        return sort_with_ranks(path_pairs, scores, reverse=True, return_rank=False)


class EmbeddingStoreRegistry:
    """Keeps record of all embedding stores available"""
    def __init__(self, save_dir, data_folder_path):
        self.save_dir = Path(save_dir)
        self.data_folder_path = data_folder_path

        self.stores: Dict[str, EmbeddingStore] = {}
        self._current_store: str = None
        self.load_state()

    @property
    def is_initialized(self):
        """Returns true if an active embedding store is chosen"""
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

    def delete_store(self, store_name=None):
        if store_name is None:
            store_name = self._current_store
        if store_name is None:
            return
        store = self.stores[store_name]
        shutil.rmtree(store.store_path)
        store.idle()
        self.stores.pop(store_name)
        self._current_store = None
        self.save_state()
