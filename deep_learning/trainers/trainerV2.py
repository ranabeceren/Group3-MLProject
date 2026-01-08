import torch
from torch import nn
from metrics.dice_score import dice_score

def train_stepV2(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               dice,
               device: torch.device):

    train_loss, train_acc, train_dice = 0, 0, 0
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

        # Dice
        dice = dice_score(y, y_pred).item()
        train_dice += dice

        """
        # Accuracy
        train_acc += accuracy_fn(
            y_true=y,
            y_pred=y_pred)
        """

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_dice /= len(data_loader)
    # train_acc /= len(data_loader)
    

    print(f"Train loss: {train_loss:.4f} | Dice Score: {train_dice:.2f}%")


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              dice,
              device: torch.device):
    
    """Performs a testing loop step on the model going over data_loader."""
    
    test_loss, test_acc, test_dice = 0, 0, 0

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

            # Calculate loss & accuracy (per batch)
            test_loss += loss_fn(test_logits, y_test)
            test_dice += dice_score(y_test, test_pred).item()

            """
            test_acc += accuracy_fn(y_true=y_test,
                                    y_pred=test_pred)
            """
        # Average test_loss & test_acc (per batch)
        test_loss /= len(data_loader)
        test_dice /= len(data_loader)
        #test_acc /= len(data_loader)
        


        # Print out what's happening

        print(f"Test loss: {test_loss:.4f} | Dice Score: {test_dice:.2f}%")
