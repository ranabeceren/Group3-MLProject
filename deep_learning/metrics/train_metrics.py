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

def pixel_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().float()
    total = y_true.numel()
    return (correct / total) * 100    

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