import torch
from torch import nn
from torch.optim import Adam

# import modules
from models.cnn_basline import BaseLineCNN
from pipelines.data_prep import data_prep
from trainers.trainer import train_model

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# paths to the patches (will be changed)
img_patches = "data/images.npy"
mask_patches = "data/masks.npy"

batch_size = 16

train_loader, test_loader, val_loader = data_prep(
    img_patches=img_patches,
    mask_patches=mask_patches,
    batch_size=batch_size
)

# prepare dataloader
model = BaseLineCNN(in_channels=3).to(device)

# define loss function
loss_fn = nn.BCEWithLogitsLoss()

# define optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# start training loop
epochs = 100
train_model(model=model,
            train_loader=train_loader,
            eval_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epochs=epochs
            )