import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# from pipelines.patch_extraction import process_city
# process_city(c, patch_size=64)

from pipelines.patch_filtering import patch_filtering
img_patches, mask_patches = patch_filtering(patches_dir="data_patches", threshold=0.6)

from pipelines.data_prep import data_prep
train_loader, val_loader, test_loader = data_prep(img_patches, mask_patches, train_split=0.7, val_split=0.15, batch_size=16)

from models.CNN import CNNBaselineModel
model = CNNBaselineModel(input_shape=4, hidden_units=32, output_shape=1).to(device)

from pipelines.training import training
training(train_loader, test_loader, val_loader, model=model, epochs=175, learning_rate=0.0005)