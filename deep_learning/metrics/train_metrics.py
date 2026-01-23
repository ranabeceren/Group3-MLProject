import torch

def binarize_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    logits: (B, 1, H, W) raw outputs from model
    returns: (B, 1, H, W) int {0,1}
    """
    probs = torch.sigmoid(logits)
    return (probs > threshold).to(torch.int32)

def binarize_targets(y: torch.Tensor) -> torch.Tensor:
    """
    y: (B, H, W) or (B, 1, H, W), possibly float
    returns: (B, 1, H, W) int {0,1}
    """
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return (y > 0.5).to(torch.int32)

def compute_pos_weight(dataloader, device):
    pos = 0
    neg = 0

    for _, y in dataloader:
        y = y.to(device)
        y = y.float()
        if y.ndim == 3:
            y = y.unsqueeze(1)

        pos += y.sum()
        neg += (1 - y).sum()

    pos_weight = neg / pos
    return pos_weight

def dice_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.float()
    y_pred = y_pred.float()

    intersection = (y_true * y_pred).sum(dim=(1,2,3))
    union = y_true.sum(dim=(1,2,3)) + y_pred.sum(dim=(1,2,3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean() * 100

def f1_score(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    F1 for binary masks (same as Dice).
    """
    y_pred = y_pred.bool()
    y_true = y_true.bool()

    tp = (y_pred & y_true).sum(dim=(1, 2, 3)).float()
    fp = (y_pred & ~y_true).sum(dim=(1, 2, 3)).float()
    fn = (~y_pred & y_true).sum(dim=(1, 2, 3)).float()

    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return f1.mean()

def iou_score(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    IoU for binary masks.
    y_pred, y_true: (B, 1, H, W) int {0,1}
    returns scalar tensor
    """
    y_pred = y_pred.bool()
    y_true = y_true.bool()

    intersection = (y_pred & y_true).sum(dim=(1, 2, 3)).float()
    union = (y_pred | y_true).sum(dim=(1, 2, 3)).float()

    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def iou_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.float()
    y_pred = y_pred.float()

    intersection = (y_true * y_pred).sum(dim=(1,2,3))
    union = y_true.sum(dim=(1,2,3)) + y_pred.sum(dim=(1,2,3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean() * 100

'''
def f1_score(y_true, y_pred, eps=1e-7):
    y_true = y_true.float()
    y_pred = y_pred.float()

    tp = (y_true * y_pred).sum(dim=(1,2,3))
    fp = (1 - y_true) * (1 - y_pred).sum(dim=(1,2,3))
    fn = (y_true * (1 - y_pred)).sum(dim=(1,2,3))

    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return f1.mean() * 100
'''