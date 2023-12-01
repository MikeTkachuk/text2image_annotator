from PIL import Image
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"
clip_model: CLIPModel = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

import time

inputs = clip_processor(images=Image.open(r"C:\Users\Michael\Downloads\tweets_data\1006776809285980160\0.jpg"),
                        return_tensors="pt")
start = time.time()
img_emb = clip_model.get_image_features(**inputs)
print(img_emb, f"elapsed: {time.time() - start}")
print(img_emb.shape)