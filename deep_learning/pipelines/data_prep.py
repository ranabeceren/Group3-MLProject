import torch
import numpy as np
import torchvision.transforms as T
from deep_learning.datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader
from deep_learning.utils.splits import train_test_val_split
from deep_learning.pipelines.patch_filtering import patch_filtering

# splits the data into the sets and creates the dataset and dataloader which are then given to the trainer

def data_prep(img_patches, mask_patches, batch_size=16, train_transform=None):

    # Get the data resulting from patch_filtering
    #img_patches, mask_patches = patch_filtering(data_dir="data_patches", threshold=threshold)

    # Split into train, test, validation sets
    train_imgs, train_masks, test_imgs, test_masks, val_imgs, val_masks = train_test_val_split(
        images=img_patches,
        masks=mask_patches,
        train=0.7,
        test=0.2,
        seed=42
    )
    # Build dataset
    train_dataset = BuildingDataset(images=train_imgs, masks=train_masks, transform=train_transform) # applies transformations only on training data
    test_dataset = BuildingDataset(images=test_imgs, masks=test_masks, transform=None)
    val_dataset = BuildingDataset(images=val_imgs, masks=val_masks, transform=None)

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