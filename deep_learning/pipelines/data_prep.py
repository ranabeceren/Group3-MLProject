import torch
import numpy as np
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader
from utils.split import train_val_test_split
#import kornia

# splits the data into the sets and creates the dataset and dataloader which are then given to the trainer

def data_prep(
    img_patches, 
    mask_patches,
    batch_size, 
    train_split,
    val_split,
    train_transform=None):

    # Get the data resulting from patch_filtering
    #img_patches, mask_patches = patch_filtering(data_dir="data_patches", threshold=threshold)

    # Split into train, test, validation sets
    train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks = train_val_test_split(
        images=img_patches,
        masks=mask_patches,
        train=train_split,
        val=val_split,
        seed=42
    )
    '''
    # Data transformations
    train_transform = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=30.0, p=0.5),
        data_keys=["input", "mask"]
    )
    '''
    # Build dataset
    train_dataset = BuildingDataset(images=train_imgs, masks=train_masks, transform=train_transform) # if the caller passes a transform → only the training data is augmented if not behaves exactly like val/test
    val_dataset = BuildingDataset(images=val_imgs, masks=val_masks, transform=None)
    test_dataset = BuildingDataset(images=test_imgs, masks=test_masks, transform=None)
    
    # Build dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False)

    return train_loader, val_loader, test_loader