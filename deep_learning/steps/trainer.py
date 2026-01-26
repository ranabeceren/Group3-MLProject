import torch
from torch import nn
from metrics.train_metrics import binarize_logits, binarize_targets, iou_score, f1_score


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):

    train_loss= 0.0
    train_acc = 0.0

    model.train()

    for batch, (X, y) in enumerate(data_loader):

        X, y = X.to(device), y.to(device)

        y = y.float()
        if y.ndim == 3:
            y = y.unsqueeze(1)

        # Forward pass
        y_logits = model(X)
        y_pred = torch.round(torch.sigmoid(y_logits)) # raw logits -> pred prob -> pred labels

        # Loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # Accuracy
        accuracy = accuracy_fn(y_pred, y)
        train_acc += accuracy.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    '''
    print(
        f"Train loss: {train_loss:.2f} | "
        f"Train accuracy: {train_acc:.2f} | "
    )
    '''
    return train_loss, train_acc

def val_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              scheduler,
              device: torch.device):
    
    """
    Performs a validation loop step on the model.
    Computes loss, accuracy, IoU and F1-score.
    """

    val_loss = 0.0
    val_acc = 0.0
    val_iou = 0.0
    val_f1 = 0.0

    model.eval()
    with torch.inference_mode():
        
        for X, y in data_loader:

            # Put data on the traget device
            X, y = X.to(device), y.to(device)

            y = y.float()
            if y.ndim == 3:
                y = y.unsqueeze(1)

            # Forward pass
            y_logits = model(X)

            # Calculate loss & accuracy (per batch)
            loss = loss_fn(y_logits, y)
            val_loss += loss.item()

            # Binary masks for metrics
            y_pred = binarize_logits(y_logits, threshold=0.3)
            y_true = binarize_targets(y)

            # IoU / F1
            val_iou += iou_score(y_pred, y_true).item()
            val_f1 += f1_score(y_pred, y_true).item()

            # Accuracy
            acc = accuracy_fn(y_pred, y_true)
            val_acc += acc.item()

    # Average test_loss & test_acc & test dice(per batch)
    val_loss /= len(data_loader)
    val_acc /= len(data_loader)
    val_iou /= len(data_loader)
    val_f1 /= len(data_loader)

    # reschedule lr by val_loss
    scheduler.step(val_loss)
    '''    
    print(
        f"Validation loss: {val_loss:.2f} | "
        f"Validation acc: {val_acc:.2f} | "
        f"Validation IoU: {val_iou:.2f} | "
        f"Validation F1: {val_f1:.2f}"
    )
    '''
    return val_loss, val_acc, val_iou, val_f1