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