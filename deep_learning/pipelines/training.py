import torch
import torch.nn as nn
from steps.trainer import train_step, val_step
from metrics.train_metrics import compute_pos_weight, print_train_time
from torchmetrics.classification import BinaryAccuracy
from timeit import default_timer as timer
from tqdm.auto import tqdm

def training(train_loader, val_loader, model, epochs, learning_rate, device):

        # Choose loss_fn, optimizer and metrics
        pos_weight = compute_pos_weight(train_loader, device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)
        accuracy_fn = BinaryAccuracy(threshold=0.5).to(device)

        # Reduce LR on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",        # because we monitor val loss
        factor=0.99,        # lr = lr * 0.99
        patience=15,        # wait 15 epochs with no improvement
        #verbose=True
        )

        history = {
        "lr": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_iou": [],
        "val_f1": []
        }

        # Measure time
        start = timer()

        # Set epochs
        epochs = epochs

        for epoch in tqdm(range(epochs)): # wraps a progression bar around epochs
                print(f"Epoch: {epoch+1}\n---------")

                # Train data
                train_loss, train_acc = train_step(
                        model=model,
                        data_loader=train_loader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        accuracy_fn=accuracy_fn,
                        device=device)

                # Validate data
                val_loss, val_acc, val_iou, val_f1 = val_step(
                        model=model,
                        data_loader=val_loader,
                        loss_fn=loss_fn,
                        accuracy_fn=accuracy_fn,
                        scheduler=scheduler,
                        device=device)

                history["lr"].append(optimizer.param_groups[0]["lr"])
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_iou"].append(val_iou)
                history["val_f1"].append(val_f1)

                print(
                        f"Train_loss={train_loss:.2f} Train_acc={train_acc:.2f} | \n"
                        f"Val_loss={val_loss:.2f} Val_acc={val_acc:.2f} "
                        f"Val_iou={val_iou:.2f} Val_f1={val_f1:.2f} | \n"
                        f"LR={optimizer.param_groups[0]['lr']}\n"
                )
        end = timer()
        train_time = print_train_time(start=start,
                                end=end,
                                device=device)
        return history