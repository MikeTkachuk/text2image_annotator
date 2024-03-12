import random
from pathlib import Path
from datetime import datetime

import tqdm
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torchvision.transforms import v2

import bitsandbytes as bnb

import wandb
from omegaconf import OmegaConf
from transformers import AutoImageProcessor, ViTMAEForPreTraining

MODEL_NAME = "facebook/vit-mae-base"
PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


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


class Dataset(TorchDataset):
    def __init__(self, samples, train=True):
        self.train = train
        self.samples = samples

        self.image_transform = v2.Compose([
            MinResize(512),
            v2.RandomChoice([SquarePad(224),
                             v2.RandomCrop(256),
                             v2.RandomCrop(380)], p=[0.4, 0.3, 0.3]),
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
        image = Image.open(sample).convert(mode="RGB")
        if self.train:
            transformed = self.image_transform(image)
        else:
            transformed = self.test_transform(image)
        return transformed


def collate_fn(samples: list):
    collated = PROCESSOR(images=samples, return_tensors="pt")
    return collated


def get_dataloaders(samples, n_test_samples, batch_size):
    train_ids = sorted(list(set([Path(s).parent.name for s in samples])))
    random.shuffle(train_ids)
    if 0 < n_test_samples < 1:
        n_test_samples = int(len(train_ids) * n_test_samples)
    test_ids = set(train_ids[:n_test_samples])
    train_ids = set(train_ids[n_test_samples:])
    train_samples = [s for s in samples if Path(s).parent.name in train_ids]
    train_dataset = Dataset(train_samples)

    test_samples = [s for s in samples if Path(s).parent.name in test_ids]
    test_dataset = Dataset(test_samples, train=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=12,
                                  collate_fn=collate_fn,
                                  pin_memory=True,
                                  prefetch_factor=3,
                                  shuffle=True,
                                  drop_last=True,
                                  persistent_workers=True
                                  )

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=0,
                                 collate_fn=collate_fn,
                                 )
    return train_dataloader, test_dataloader


def train():
    run_name = "faster_steps5"
    config = OmegaConf.create(
        {"n_epochs": 120,
         "batch_size": 56,
         "learning_rate": 8e-5,
         "accumulate_steps": 2,
         "test_samples": 0.05,
         "weight_decay": 0.0,
         "agc_lambda": 0.05,
         "seed": 42,
         }
    )
    wandb.init(
        project="clip-annotator",
        config=dict(config),
        name=run_name,
        tags=["mae_pretrain"],
        mode="online"
    )
    seed_everything(config.seed)

    data_dir = Path(r'C:/Users/Michael/Downloads/tweets_data')
    all_files = [str(f) for f in data_dir.rglob("*") if f.suffix in [".jpeg", ".jpg", ".png"] and f.stat().st_size]
    train_loader, test_loader = get_dataloaders(all_files, config.test_samples, config.batch_size)
    steps_per_epoch = len(train_loader.dataset) // config.batch_size // config.accumulate_steps
    save_every = 5
    checkpoint_dir = Path(r"C:\Users\Michael\PycharmProjects\text2image_annotator"
                          ) / f"embedders/{run_name}_{datetime.now().strftime('%m-%d_%H-%M')}"
    model: ViTMAEForPreTraining = ViTMAEForPreTraining.from_pretrained(MODEL_NAME, device_map=device)
    optim = bnb.optim.AdamW8bit(params=model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optim,
                                                       max_lr=config.learning_rate,
                                                       total_steps=steps_per_epoch * config.n_epochs,
                                                       pct_start=0.1
                                                       )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optim,
                                                     start_factor=1, end_factor=0.01,
                                                     total_iters=steps_per_epoch * config.n_epochs)
    for epoch in tqdm.tqdm(range(config.n_epochs), desc="Epochs:"):
        model.train()
        to_log = {}
        for i, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            if i > len(train_loader) // config.accumulate_steps * config.accumulate_steps:
                break
            batch = batch.to(device)
            if i == 0:
                with torch.no_grad():
                    model.eval()
                    to_log["train_viz"] = wandb.Image(visualize(model, batch, list(range(9))))
                    model.train()
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss
                wandb.log({"loss": loss.item(),
                           "lr": lr_scheduler.get_last_lr()[0],
                           "epoch": epoch,
                           })  # log before scaling
                loss /= config.accumulate_steps
            loss.backward()
            if (i + 1) % config.accumulate_steps == 0:
                adaptive_gradient_clipping(model.parameters(), lam=config.agc_lambda)
                optim.step()
                lr_scheduler.step()

        # eval
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(device)
                if i == 0:
                    to_log["eval_viz"] = wandb.Image(visualize(model, batch, list(range(9))))
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    output = model(**batch)
                    eval_losses.extend([output.loss.item()] * len(batch.pixel_values))
            to_log["test_loss"] = np.mean(eval_losses)
            wandb.log(to_log)

        # checkpoint
        if epoch % save_every == 0:
            model.save_pretrained(checkpoint_dir)
    model.save_pretrained(checkpoint_dir)


@torch.no_grad()
def adaptive_gradient_clipping(parameters, lam=0.15):
    for param in parameters:
        ratio = torch.abs(param.grad) / (torch.abs(param) + 1E-3)
        param.grad = torch.where(ratio > lam, lam * param.grad / ratio, param.grad)


def visualize(model, inputs, ids, mode=0):
    # forward pass
    outputs = model(**inputs)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', inputs.pixel_values).detach().cpu()

    # masked image
    # im_masked = x * (1 - mask)

    # im_paste = x * (1 - mask) + y * mask

    if mode == 0:
        images = y
    elif mode == 1:
        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask
        images = im_paste
    imagenet_mean = np.array(PROCESSOR.image_mean)
    imagenet_std = np.array(PROCESSOR.image_std)
    square_size = int(np.ceil(np.sqrt(len(ids))))
    img_size = images.shape[1]
    canvas = torch.zeros((img_size * square_size, img_size * square_size, 3), dtype=torch.uint8)
    for i in range(len(ids)):
        row = i // square_size
        column = i - row * square_size
        image = torch.clip((images[ids[i]] * imagenet_std + imagenet_mean) * 255, 0, 255).int()
        canvas[row * img_size:(row + 1) * img_size, column * img_size:(column + 1) * img_size] = image

    return canvas.numpy()


if __name__ == "__main__":
    train()
