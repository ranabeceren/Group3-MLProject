def dice_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.float()
    y_pred = y_pred.float()

    intersection = (y_true * y_pred).sum(dim=(1,2,3))
    union = y_true.sum(dim=(1,2,3)) + y_pred.sum(dim=(1,2,3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean() * 100