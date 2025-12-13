import torch
from torch import nn

'''
Dice Formula:

'''
class DiceLoss(nn.Module): # defining our own loss class which inherits from nn.Module
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth # prevents division by 0 in case tensor is empty or no overlapping is happening

    def forward(self, preds, target):
        # Sigmoid converts our logits into probabilities between 0 and 1
        preds = torch.sigmoid(preds)

        # Flatten tensor from (B, H, W) to (B*H*W) to be able to use the dice formula with them
        preds_flat = preds.view(-1)
        targets_flat = target.view(-1)

        # Calculates the correct predicted pixels (overlapping/intersection)
        intersection = (preds_flat * targets_flat).sum()

        # Dice loss formula (the better the overlapping, the smaller is the loss)
        return 1 - ((2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth))



class BCEDiceLoss(nn.Module): # combines Dice loss and BCE loss to one loss function
    def __init__(self, alpha=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss() # BCE: binary classification for pixels (0 or 1)
        self.dice = DiceLoss() # dice loss for overlapping
        self.alpha = alpha # weighting between the BCE and Dice loss (so 0.5 is equally weighted)

    def forward(self, preds, targets): # calculates both losses and retuns the average
        # calculate BCE Loss
        bce_loss = self.bce(preds, targets)

        # calculate Dice Loss
        dice_loss = self.dice(preds, targets)

        # combine the loss functions
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

