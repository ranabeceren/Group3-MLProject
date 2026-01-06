import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

# import from metrics (doesnt exist yet)
from utils import dice, iou, losses

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train() # change to train mode
    epoch_loss, epoch_dice, epoch_iou = 0.0, 0.0, 0.0

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
        epoch_dice += dice(preds, masks)
        epoch_iou += iou(preds, masks)

    # average metrics
    n_batches = len(dataloader)

    return ( # calculate the average of all batches
        epoch_loss / n_batches,
        epoch_dice / n_batches,
        epoch_iou / n_batches
    )

def eval_step(model, dataloader, loss_fn, device):
    model.eval() # switches model into evaluation mode
    epoch_loss, epoch_dice, epoch_iou = 0.0, 0.0, 0.0

    with torch.no_grad(): # prevents gradient-calculation to save space
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            preds = model(images)

            preds = preds.squeeze(1)
            masks = masks.squeeze(1)

            loss = loss_fn(preds, masks)

            epoch_loss += loss.item()
            epoch_dice += dice(preds, masks)
            epoch_iou += iou(preds, masks)
    n_batches = len(dataloader)

    return (
        epoch_loss / n_batches,
        epoch_dice / n_batches,
        epoch_iou / n_batches
    )

def train_model(model, train_loader, eval_loader, loss_fn, optimizer, device, epochs):

    torch.manual_seed(42) # seed for reproducability
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)

    model.to(device)  # model to CPU/GPU

    best_iou = 0.0

    for epoch in range(epochs):
        print(f"\n----- Epoch {epoch+1}/epochs -----")

        # carries out the training and evaluation steps
        train_loss, train_dice, train_iou = train_step(
            model, train_loader, loss_fn, optimizer, device
        )

        eval_loss, eval_dice, eval_iou = eval_step(
            model, eval_loader, loss_fn, device
        )

        print(
            f"Train Loss: {train_loss: .4f} | "
            f"Train Dice: {train_dice: .4f} | IoU: {train_iou: .4f}"
        )

        print(
            f"Eval Loss: {eval_loss: .4f} | "
            f"Eval Dice: {eval_dice: .4f} | IoU: {eval_iou: .4f}"
        )

        # automatically saves the model with the best IoU
        if eval_iou > best_iou:
            best_iou = eval_iou
            torch.save(model.state_dict(), f"best_model.pth")
            print("-> Best model saved!")