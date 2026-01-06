import torch
from torch.utils.data import Dataset


class BuildingDataset(Dataset):
    """
    Custom Dataset class for building segmentation.
    (Inherits form PyTorch's Dataset class.)
    """
    def __init__(self,
                 images,
                 masks,
                 transforms=None):
        """
        Args:
            images (np.ndarray): (N, C, H, W)
            masks (np.ndarray): (N, H, W)
            transforms (optional): augmentations 
        """

        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,
                    idx):
        X = torch.tensor(self.images[idx], dtype=torch.float32)
        y = torch.tensor(self.masks[idx], dtype=torch.float32)

        if self.transforms:
            X, y = self.transforms(X, y)

        return X, y
