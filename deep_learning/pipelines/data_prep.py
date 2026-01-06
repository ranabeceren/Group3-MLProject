import torch
import numpy as np
from datasets.build_dataset import BuildDataset
from torch.utils.data import DataLoader
from utils.splits import train_test_val_split
from pipelines.patch_filtering import patch_filtering

def data_prep(img_patches, mask_patches, batch_size=16):

    # Get the data resulting from patch_filtering
    img_patches, mask_patches = patch_filtering(data_dir="data_patches", threshold=0.75)
    # Split into train, test, validation sets
    train_imgs, train_masks, test_imgs, test_masks, val_imgs, val_masks = train_test_val_split(
        images=img_patches,
        masks=mask_patches,
        train=0.7,
        test=0.2,
        seed=42
    )
    # Build dataset
    train_dataset = BuildDataset(images=train_imgs, masks=train_masks)
    test_dataset = BuildDataset(images=test_imgs, masks=test_masks)
    val_dataset = BuildDataset(images=val_imgs, masks=val_masks)

    # Build dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False)
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False)
    return train_loader, test_loader, val_loader