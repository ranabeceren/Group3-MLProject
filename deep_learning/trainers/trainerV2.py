import torch
from torch import nn

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               dice_fn,
               iou_fn,
               device: torch.device):

    train_loss, train_acc, train_dice, train_io = 0, 0, 0, 0
    model.train()

    for batch, (X, y) in enumerate(data_loader):

        X, y = X.to(device), y.to(device)

        # (Look at this again)
        y = y.float()
        if y.ndim == 3:
            y = y.unsqueeze(1)

        # Forward pass
        y_logits = model(X)
        y_pred = torch.round(torch.sigmoid(y_logits)) # raw logits -> pred prob -> pred labels

        # Loss
        loss = loss_fn(y_logits, y)
        train_loss += loss

        # Accuracy
        accuracy = accuracy_fn(y_pred, y)
        train_acc += accuracy

        # Dice Score
        dice = dice_fn(y_logits, y)
        train_dice += dice

        # IoU Score
        iou = iou_fn(y_logits, y)
        train_io += iou

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    train_dice /= len(data_loader)
    train_io /= len(data_loader)
    

    print(
        f"Train loss: {train_loss:.2f} | "
        f"Train accuracy: {train_acc:.2f} | "
        f"Train dice: {train_dice:.2f} | "
        f"Train IO: {train_io:.2f}"
    )


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              dice_fn,
              iou_fn,
              scheduler,
              device: torch.device):
    
    """Performs a testing loop step on the model going over data_loader."""
    
    test_loss, test_acc, test_dice, test_io = 0, 0, 0, 0

    model.eval()
    with torch.inference_mode():
        
        for X_test, y_test in data_loader:

            # Put data on the traget device
            X_test, y_test = X_test.to(device), y_test.to(device)

            # (Look at this again)
            y_test = y_test.float()
            if y_test.ndim == 3:
                y_test = y_test.unsqueeze(1)

            # Forward pass
            test_logits = model(X_test)
            test_pred = torch.round(torch.sigmoid(test_logits)) 

            # Calculate loss
            loss = loss_fn(test_logits, y_test)
            test_loss += loss

            # accuracy (per batch)
            accuracy = accuracy_fn(test_pred, y_test)
            test_acc += accuracy

            # test dice & test iou
            test_dice += dice_fn(test_logits, y_test)
            test_io += iou_fn(test_logits, y_test)
            
        # Average test_loss & test_acc & test dice(per batch)
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        test_dice /= len(data_loader)
        test_io /= len(data_loader)

        # reschedule lr by val_loss
        scheduler.step(test_loss.item())

        # Print out what's happening
        print(
            f"Test loss: {test_loss:.2f} |"
            f"Test accuracy: {test_acc:.2f} |"
            f"Test dice: {test_dice:.2f} |"
            f"Test IO: {test_io:.2f}"
        )
