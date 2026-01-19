import torch
from torch.utils.data import Dataset
import numpy as np

class BuildingDataset(Dataset):
    """
    Custom Dataset class for building segmentation.
    (Inherits form PyTorch's Dataset class.)
    """
    def __init__(self, images, masks, transform=None):
        """
        Args:
            images (np.ndarray): (N, C, H, W)
            masks (np.ndarray): (N, H, W)
            transform: augmentations
        """

        self.img_patches = images
        self.mask_patches = masks
        self.transform = transform

    def __len__(self):
        return len(self.img_patches)
    
    def __getitem__(self, idx):

        X = torch.tensor(self.img_patches[idx], dtype=torch.float32)
        y = torch.tensor(self.mask_patches[idx], dtype=torch.float32)

        if y.ndim == 2:
            y = y.unsqueeze(0)

        # transform image (aguments and ToTensor)
        if self.transform:
            X_batch = X.unsqueeze(0)
            y_batch = y.unsqueeze(0)

            X_aug, y_aug = self.transform(X_batch, y_batch)

            X = X_aug.squeeze(0)
            y = y_aug.squeeze(0)

        return X, y
