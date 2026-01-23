import torch
import torch.nn as nn
from steps.tester import test_step
from torchmetrics.classification import BinaryAccuracy

# Test data
def evaluater(model, test_loader, device):
    accuracy_fn = BinaryAccuracy(threshold=0.5).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    results = test_step(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )
    print(results)
