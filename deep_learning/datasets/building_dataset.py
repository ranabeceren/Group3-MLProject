import torch
from torch.utils.data import Dataset


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

        X = self.img_patches[idx] # keeping as numpy array for now
        y = self.mask_patches[idx]

        #X = torch.tensor(self.img_patches[idx], dtype=torch.float32)
        #y = torch.tensor(self.mask_patches[idx], dtype=torch.float32)

        # transform image (aguments and ToTensor)
        if self.transform:
            X  = self.transform(X)

        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        return X, y
