from datasets.build_dataset import build_dataset
from torch.utils.data import DataLoader

def build_dataloaderV2(images, masks, batch_size=32, shuffle=None):
    dataset = BuildDataset(images, masks, transforms=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader
