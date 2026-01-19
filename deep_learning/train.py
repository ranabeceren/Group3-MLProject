import os
import sys
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torchvision.transforms as T

# import modules
from deep_learning.models.UNetModel import UNet
from deep_learning.pipelines.data_prep import data_prep
from deep_learning.trainers.trainer import train_model
from deep_learning.pipelines.patch_extraction import process_city
from deep_learning.pipelines.patch_filtering import patch_filtering

'''
Settings
'''
# paths to the data
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
PATCH_DIR = "data/patches"
PATCH_SIZE = 64

BATCH_SIZE = 16
EPOCHS = 1000
LR = 0.0005
PATCH_THRESHOLD = 0.6

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

'''
Patch extraction
'''
CITIES = ["amsterdam", "barcelona", "berlin", "brisbane",
    "cairo", "darmstadt"]

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
img_patches,mask_patches = patch_filtering(patches_dir="data_patches",
                                           threshold=PATCH_THRESHOLD)

#print(f"{len(img_patches)} patches remaining after filtering")
'''
Data transformation
'''
train_transform = None
# T.Compose([
#    T.RandomHorizontalFlip(p=0.5), #rotations on the tensors
 #   T.RandomVerticalFlip(p=0.5)
#])

val_transform = None #in case we need it later

'''
Data preparation
'''
print("Creating dataloaders")
train_loader, test_loader, val_loader = data_prep(
    img_patches=img_patches,
    mask_patches=mask_patches,
    batch_size=BATCH_SIZE,
    train_transform=train_transform
)

print(f"Tain/Val/Test sizes: {len(train_loader.dataset)}, {len(test_loader.dataset)}, {len(val_loader.dataset)}")

'''
Model, Loss, Optimizer
'''
model = UNet(in_channels=4).to(device)
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