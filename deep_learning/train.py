import os
import torch
import numpy as np
from torch import nn
from torch.optim import Adam

# import modules
from .models.cnn_basline import BaseLineCNN
from .pipelines.data_prep import data_prep
from .trainers.trainer import train_model
from .pipelines.patch_extraction import process_city
from .pipelines.patch_filtering import patch_filtering

'''
Settings;
'''
# paths to the data
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
PATCH_DIR = "data/patches"
PATCH_SIZE = 64

BATCH_SIZE = 16
EPOCHS = 100
LR = 0.001
PATCH_THRESHOLD = 0.01

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

'''
Patch extraction
'''
CITIES = [
    "amsterdam", "barcelona", "berlin", "brisbane",
    "cairo", "darmstadt", "lisbon", "madrid",
    "melbourne", "mexico_city", "porto"
]

print("Processing all cities and extracting patches")
for city in CITIES:
    city_patches_img, city_patches_mask = process_city(city, PATCH_SIZE)

    #if city_patches_mask is not None:
     #   all_img_patches.extend(city_patches_img)
      #  all_mask_patches.extend(city_patches_mask)

#print(f"Number of patches:", len(all_img_patches))

'''
Patch filtering
'''

print(f"Filtering patches with threshold: {PATCH_THRESHOLD}")
img_patches,mask_patches = patch_filtering(data_dir="data_patches",
                                           threshold=PATCH_THRESHOLD)

#print(f"{len(img_patches)} patches remaining after filtering")

'''
Data preparation
'''
print("Creating dataloaders")
train_loader, test_loader, val_loader = data_prep(
    img_patches=img_patches,
    mask_patches=mask_patches,
    batch_size=BATCH_SIZE,
    threshold=PATCH_THRESHOLD
    )

print(f"Tain/Val/Test sizes: {len(train_loader.dataset)}, {len(test_loader.dataset)}, {len(val_loader.dataset)}")

'''
Model, Loss, Optimizer
'''
model = BaseLineCNN(in_channels=3).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LR)

'''
Training
'''
print("Starting training")

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    epochs=EPOCHS)

print("Finished training")