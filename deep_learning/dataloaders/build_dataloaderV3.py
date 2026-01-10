import numpy as np
import torch
from torch.ao.nn.quantized.functional import threshold
from torch.utils.data import DataLoader
from datasets.building_dataset import BuildingDataset
from utils.splits import train_test_val_split
from pipelines.patch_filtering import patch_filtering

from deep_learning.dataloaders.build_dataloader import train_dataset

def create_dataloaders(
        images, # raw image patches are numpy arrays
        masks,
        batch_size=16,
        num_workers=4,
        transforms=None,
        patch_threshold=0.7,
        seed=42
):

    '''
    Creates train/val/test DataLoader fpr budilding segemntation
    returns:
    train_loader, val_loader, test_loader
    '''

    # filter patches with not enough buidlings
    masks, images = patch_filtering(masks, images, threshold=patch_threshold)

   # split in train/test sets/val
    train_images, test_images, val_images, train_masks, test_masks, val_masks = train_test_val_split(
        images, masks, train=0.7, test=0.2, seed=seed)

    # create Datasets
    train_dataset = BuildingDataset(train_images, train_masks, transforms=transforms)
    val_dataset = BuildingDataset(val_images, val_masks, transforms=None)
    test_dataset = BuildingDataset(test_images, test_masks, transforms=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # num of parallel worker threads
        pin_memory=torch.cuda.is_available()   # when we use GPU instead of CPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
        )

    print(f"Dataloaders ready: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    return train_loader, val_loader, test_loader