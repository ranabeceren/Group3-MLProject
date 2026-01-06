from locale import normalize

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class Buildingdataset(Dataset): # creating custom Dataset (expects us to implement len and getitem)
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths # List of all Image-Datapaths
        self.mask_paths = mask_paths # List of all mask-datapaths
        self.transform = transform # in case of agumentation of the data
        self.normalize = normalize

    def __len__(self):
        return len(self.image_paths) # returns the amount of samples

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]), dtype=np.float32) # reads the TIFF data

        if self.normalize:
            image = image  / 255.0 # if RGB only takes first channel

        mask = np.array(Image.open(self.mask_paths[idx]), dtype=np.float32) # reads in the mask TIFF data
        if mask.ndim == 3: # in case mask is RGB
            mask = mask[:, :, 0] # only takes the first channel


        # create tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Channel first
        if image.ndim == 3:
            image = image.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        else:
            image = image.unsqueeze(0) # if there is greyscale

        mask = mask.unsqueeze(0) # masks always have (1, H, W) format

        # Optional Transformation
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask