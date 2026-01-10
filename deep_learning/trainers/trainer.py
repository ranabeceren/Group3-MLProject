import torch
from tqdm.auto import tqdm
from dataloaders.build_dataloader import create_dataloaders
import torch.nn.functional as F

# import from metrics (doesnt exist yet)
#from utils import dice, iou, loss_function

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train() # change to train mode
    epoch_loss = 0.0

    # loop through all the batches of the dataloader
    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device) # device agnostic code

        optimizer.zero_grad() # deletes gradient of the previous step

        preds = model(images) # prediction of the model

        preds = preds.squeeze(1) # removes canal dim for loss calculation (B,1,H,W) -> (B,H,W)
        masks = masks.squeeze(1)

        loss = loss_fn(preds, masks) # calculates the loss

        loss.backward() # Backpropagation
        optimizer.step() # updates the weights

        # accumulates loss and metrics
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def eval_step(model, dataloader, loss_fn, device):
    model.eval() # switches model into evaluation mode
    epoch_loss = 0.0

    with torch.no_grad(): # prevents gradient-calculation to save space
        for images, masks in tqdm(dataloader, desc="validation", leave=False):
            images, masks = images.to(device), masks.to(device)

            preds = model(images)

            preds = preds.squeeze(1)
            masks = masks.squeeze(1)

            loss = loss_fn(preds, masks)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):

    torch.manual_seed(42) # seed for reproducability
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)

    model.to(device)  # model to CPU/GPU

    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n----- Epoch {epoch+1}/epochs -----")

        # carries out the training and evaluation steps
        train_loss = train_step(
            model, train_loader, loss_fn, optimizer, device
        )

        train_loss = train_step(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_step(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss: .4f} | Eval Loss: {val_loss: .4f}")

        # automatically saves the model with the best IoU
       # if eval_iou > best_iou:
        #    best_iou = eval_iou
         #   torch.save(model.state_dict(), f"best_model.pth")
         #   print("-> Best model saved!")