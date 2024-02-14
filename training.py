import json
from pathlib import Path
from typing import Union, List

import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision.transforms import v2

import bitsandbytes as bnb
from transformers import CLIPModel, CLIPProcessor, BatchFeature

import wandb
from omegaconf import OmegaConf

MODEL_NAME = "openai/clip-vit-large-patch14"
PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return v2.functional.pad(image, padding, 0, 'constant')


class Dataset(TorchDataset):
    def __init__(self, session_config, subsample=None, train=True):
        self.train = train
        self.data_pool: dict = session_config["data"]
        self.samples = [f for f in self.data_pool.keys()]
        if subsample is not None:
            self.samples = [s for s in self.samples if Path(s).parent.name in subsample]
        self.samples = [s for s in self.samples if "delete" not in self.data_pool[s]["tags"]]

        self.image_transform = v2.Compose([
            SquarePad(),
            v2.Resize(512),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=[0, 30]),
            v2.RandomApply([
                v2.ColorJitter(brightness=0.2, saturation=0.2, hue=0.2, contrast=0.2)
            ],
                p=0.7),
        ])

        self.test_transform = SquarePad()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # image
        image = Image.open(sample)
        if self.train:
            transformed = self.image_transform(image)
        else:
            transformed = self.test_transform(image)

        # label
        metadata = self.data_pool[sample]
        tags = metadata["tags"]
        sampled_meta = []
        if self.train:
            if np.random.random() < 0.7 and metadata["prompt"]:
                sampled_meta.append(metadata["prompt"])
            if len(tags) >= 1:
                key = np.random.random()
                if key < 0.4:  # 40% learn single tags
                    n_tags = 1
                else:
                    n_tags = np.random.randint(1, len(tags) + 1)
                if key > 0.8:  # 20% use all tags
                    n_tags = len(tags)
                sampled_meta.extend(np.random.choice(tags, size=n_tags, replace=False))
            text = ",".join(sampled_meta)
        else:
            text = ",".join([metadata["prompt"]] + metadata["tags"])

        # tokenized
        inputs = PROCESSOR(text=text, images=transformed, return_tensors="pt", padding="max_length")
        full_label = ([metadata["prompt"]] + metadata["tags"], sampled_meta)
        return inputs, full_label


def collate_fn(samples: list):
    collated = {k: torch.cat([sample[0][k] for sample in samples]) for k in samples[0][0]}
    collated = BatchFeature(data=collated)
    target = torch.zeros((len(samples), len(samples)), dtype=torch.float32)

    set_like_meta = [(set(sample[1][0]), set(sample[1][1])) for sample in samples]
    for i_image in range(target.shape[0]):
        for i_text in range(target.shape[1]):
            text_label = set_like_meta[i_text][1]
            img_label = set_like_meta[i_image][0]
            if img_label.intersection(text_label) == text_label and text_label:
                target[i_text, i_image] = 1
    return collated, target, [s[1][0] for s in samples]


def get_dataloaders(data_config, n_test_samples, batch_size):
    train_ids = list(set([Path(s).parent.name for s in data_config["data"].keys()]))
    test_ids = set(train_ids[:n_test_samples])
    train_test_ids = set(train_ids[n_test_samples:n_test_samples + 30])
    train_ids = set(train_ids[n_test_samples:])

    train_dataset = Dataset(data_config, subsample=train_ids)

    test_dataset = Dataset(data_config, subsample=test_ids, train=False)
    train_test_dataset = Dataset(data_config, subsample=train_test_ids, train=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=1,
                                  collate_fn=collate_fn,
                                  pin_memory=True,
                                  prefetch_factor=1,
                                  shuffle=True,
                                  )

    train_test_dataloader = DataLoader(train_test_dataset,
                                       batch_size=1,
                                       num_workers=0,
                                       collate_fn=collate_fn,
                                       )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=0,
                                 collate_fn=collate_fn,
                                 )
    return train_dataloader, train_test_dataloader, test_dataloader


def train(model_name=MODEL_NAME):
    config = OmegaConf.create(
        {"n_epochs": 50,
         "batch_size": 4,
         "learning_rate": 1e-6,
         "accumulate_steps": 16,  # TODO store embedding from last steps for full rank matrix loss
         "test_samples": 40,
         }
    )
    wandb.init(
        project="clip-annotator",
        config=dict(config),
        name="smoler_lr"
    )
    # config

    data_config = json.load(open(r"./output/tweets_data.json"))
    clip_model: CLIPModel = CLIPModel.from_pretrained(model_name, device_map=device)
    train_dataloader, train_test_dataloader, test_dataloader = get_dataloaders(data_config,
                                                                               config.test_samples,
                                                                               config.batch_size
                                                                               )
    steps_per_epoch = len(train_dataloader.dataset) // config.batch_size // config.accumulate_steps
    eval_results = evaluate(clip_model, data_config, [train_test_dataloader, test_dataloader])
    print("Before training: ", eval_results)
    wandb.log({"f1_train": eval_results[0]["score"], "f1": eval_results[1]["score"]})
    optim = bnb.optim.AdamW8bit(params=clip_model.parameters(), lr=config.learning_rate, weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optim,
                                                       max_lr=config.learning_rate,
                                                       total_steps=steps_per_epoch * config.n_epochs,
                                                       pct_start=0.2
                                                       )
    for epoch in range(config.n_epochs):
        clip_model.train()
        for i, (data, target, _) in tqdm.tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}: "):
            if i > steps_per_epoch * config.accumulate_steps:
                break
            data = data.to(device)
            target = target.to(device)
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                txt_emb, img_emb = clip_model(**data, return_loss=False)[2:4]
                logits = torch.matmul(txt_emb, img_emb.t())
                loss = torch.nn.BCEWithLogitsLoss()(logits.flatten(), target.flatten())
                loss /= config.accumulate_steps
                wandb.log({"loss": loss.item()})
            loss.backward()
            if (i + 1) % config.accumulate_steps == 0:
                optim.step()
                lr_scheduler.step()
        # evaluate
        clip_model.eval()
        eval_results = evaluate(clip_model, data_config, [train_test_dataloader, test_dataloader])
        wandb.log({"f1_train": eval_results[0]["score"], "f1": eval_results[1]["score"]})

    wandb.finish()


def evaluate(model: CLIPModel, data_config, dataloaders):
    annotator = get_annotator(model, data_config)
    results = []
    for dataloader in dataloaders:
        preds = []
        for data, _, label in dataloader:
            if len(label[0]) < 2:
                print("Empty label in evaluation", label)
                continue
            pred = annotator(data.to(device)).flatten()
            if not len(pred):
                print("Empty pred in evaluation")
                continue

            preds.append((pred.cpu().numpy(), label[0][1:]))

        thresh_space = np.linspace(0.1, 0.9, 10)
        thresh_selection = []
        for thresh in tqdm.tqdm(thresh_space):
            thresh = np.quantile([p[0] for p in preds], q=thresh)
            scores = []
            for p, l in preds:
                p_tags = [t for t, k in zip(data_config["tags"], p) if k > thresh]
                if not p_tags or not l:
                    continue
                scores.append(tag_score(p_tags, l))
            thresh_selection.append(np.mean(scores))

        best_id = np.nanargmax(thresh_selection)
        print(thresh_selection)
        results.append({"score": thresh_selection[best_id], "thresh": thresh_space[best_id]})
    return results


@torch.no_grad()
def get_annotator(model: CLIPModel, data_config, sampling_mode="classification"):
    if sampling_mode == "classification":
        tag_embeddings = model.get_text_features(
            **PROCESSOR(text=data_config["tags"], padding=True, return_tensors="pt").to(device)
        )
        tag_embeddings = tag_embeddings / tag_embeddings.norm(p=2, dim=-1, keepdim=True)

        @torch.no_grad()
        def func(inp, thresh=0):
            img_embeddings = model.get_image_features(pixel_values=inp["pixel_values"])
            img_embeddings = img_embeddings / img_embeddings.norm(p=2, dim=-1, keepdim=True)
            scores = torch.matmul(img_embeddings, tag_embeddings.t())[0]
            return scores

        return func


def tag_score(pred, true, mode="f1"):
    correct = set(pred).intersection(set(true))
    if mode == "p":
        return len(correct) / len(pred)
    if mode == "r":
        return len(correct) / len(true)
    if mode == "f1":
        p = len(correct) / len(pred)
        r = len(correct) / len(true)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@torch.no_grad()
def embedding_tree_train(embedder: CLIPModel, data_aug_passes=4):
    data_config = json.load(open(r"./output/tweets_data.json"))
    train_dl, _, test_dl = get_dataloaders(data_config, 40, 16)
    train_emb, test_emb = [], []
    label_train, label_test = [], []
    for i in range(data_aug_passes):
        for data, _, label in tqdm.tqdm(train_dl):
            data.to(device)
            train_emb.extend(embedder(**data, return_dict=True)["image_embeds"].cpu().numpy().tolist())
            label_train.extend(label)

    for data, _, label in tqdm.tqdm(test_dl):
        data.to(device)
        test_emb.append(embedder(**data, return_dict=True)["image_embeds"][0].cpu().numpy())
        label_test.append(label[0])

    X_train = np.stack(train_emb, axis=0)
    X_test = np.stack(test_emb, axis=0)
    y_full = {}
    results = {}
    for tag in data_config["tags"]:
        y_train = np.array([tag in l for l in label_train])
        y_test = np.array([tag in l for l in label_test])
        y_full[tag] = {
            "y_train": y_train,
            "y_test": y_test
        }

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        f1_ = f1_score(y_test, model.predict(X_test))
        results[tag] = {
            "model": model,
            "f1_score": f1_,
            "imbalance": [sum(y_train)/len(y_train), sum(y_test)/len(y_test)]
        }

    return results


if __name__ == "__main__":
    clip_model: CLIPModel = CLIPModel.from_pretrained(MODEL_NAME, device_map=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    img_src = PROCESSOR(images=Image.open(r"C:\Users\Michael\Downloads\tweets_data\623707697020403712\0.jpg"),
                        return_tensors="pt")
    img_dst: BatchFeature = PROCESSOR(images=Image.open(r"C:\Users\Michael\Downloads\tweets_data\753599157177167872\0.jpg"),
                                      return_tensors="pt")
    dst_emb = clip_model.get_image_features(img_dst["pixel_values"].to(device))

    w = 0.3
    src_tensor = (w * img_src["pixel_values"] + (1 - w) * img_dst["pixel_values"]).to(device)
    src_tensor.requires_grad = True
    adm = torch.optim.SGD([src_tensor], lr=0.00003, momentum=0.0)
    lr = 3
    for i in range(10000):
        emb = clip_model.get_image_features(src_tensor)
        loss = torch.abs(emb - dst_emb).mean()
        loss.backward()
        print(loss.item())
        adm.step()
        if i % 200 == 0:
            viz = src_tensor.detach().cpu().numpy()[0].transpose(1,2,0)
            viz = viz * 0.43 + 0.5
            plt.imshow(viz)
            plt.show()
    # with torch.no_grad():
        #     src_tensor -= src_tensor.grad * lr
        #     src_tensor.grad = None
        # if i % 200 == 0:
        #     viz = src_tensor.detach().cpu().numpy()[0].transpose(1,2,0)
        #     inp = input()
        #     if inp:
        #         lr = float(inp)
    # res = embedding_tree_train(clip_model)
    # bar = [(n,res[n]["f1_score"]) for n in res if res[n]["f1_score"] > 0]
    # bar = sorted(bar, key=lambda x: x[1])
    # bar_names, bar_values = [i for i in zip(*bar)]
    # plt.bar(x=bar_names, height=bar_values)
    # plt.xticks(rotation=30, ha='right')
    # plt.show()