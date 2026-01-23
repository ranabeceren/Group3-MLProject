import torch
import os
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
'''
CITIES = [
    "amsterdam", "barcelona", "berlin", "brisbane",
    "cairo", "darmstadt", "lisbon", "madrid",
    "melbourne", "mexico_city", "porto"
]

from pipelines.patch_extraction import process_city
for city in tqdm(CITIES):
    process_city(city, patch_size=32)
'''
from pipelines.patch_filtering import patch_filtering
img_patches, mask_patches = patch_filtering(patches_dir="data_patches", threshold=0.6)

from pipelines.data_prep import data_prep
train_loader, val_loader, test_loader = data_prep(img_patches, mask_patches, batch_size=16, train_split=0.7, val_split=0.15)

from models.cnn_model import CNNBaselineModel
model = CNNBaselineModel(input_shape=4, hidden_units=32, output_shape=1).to(device)

from pipelines.training import training
training(train_loader, val_loader, model=model, epochs=150, learning_rate=0.0005)

from pipelines.evaluater import evaluater
model_results = evaluater(model, test_loader, device)

