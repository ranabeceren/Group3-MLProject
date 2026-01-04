import torch
from torch.utils.data import DataLoader
from datasets.build_dataset import BuildDataset
Mode = "Test"

if Mode == "Test":

    #Test Dataloader 

    '''
    Using the Dataset from building_dataset
    '''

    train_dataset = BuildingDataset(
        images=train_images,
        masks=train_masks,
        transforms=train_transforms
        )
    '''
    Set the Dataloader for training
    '''

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,      #which size is usefull?
        shuffle=True,
        num_workers=0       #num of parralel worker threads, no parallel processing at first
        #pin_memory=True    # when we use GPU instead of CPU

    )
    '''
    Test the Dataloader 
    '''

    images, masks = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
    print("Images dtype:", images.dtype)
    print("Masks dtype:", masks.dtype)

elif Mode == "Train":
  
    def create_dataloaders(
        train_images,
        train_masks,
        val_images,
        val_masks,
        batch_size=16,
        num_workers=0,
        transforms=None
    ):
        train_dataset = BuildingDataset(
            train_images, 
            train_masks, 
            transforms=transforms
        )

        val_dataset = BuildingDataset(
            val_images, 
            val_masks, 
            transforms=None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
           # pin_memory=torch.cuda.is_available() pin_memory=True when we use GPU instead of CPU
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
          #  pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader    
