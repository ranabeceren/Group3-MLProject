import torch
import torch.nn as nn
from steps.tester import test_step
from torchmetrics.classification import BinaryAccuracy
from metrics.train_metrics import compute_pos_weight

# Test data
def testing(model, model_name, test_loader, device):
    pos_weight = compute_pos_weight(test_loader, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    accuracy_fn = BinaryAccuracy(threshold=0.5).to(device)

    test_loss, test_acc, test_iou, test_f1 = test_step(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )
    print(
        f"Test_loss={test_loss:.2f} Test_acc={test_acc:.2f} | \n"
        f"Test_iou={test_iou:.2f} Test_f1={test_f1:.2f} "
    )
    return {
        "model_name": model.__class__.__name__,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_iou": test_iou,
        "test_f1": test_f1
    }
