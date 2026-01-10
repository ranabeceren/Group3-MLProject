def pixel_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().float()
    total = y_true.numel()
    return (correct / total) * 100    
