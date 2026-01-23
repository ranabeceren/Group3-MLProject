import torch
from tqdm.auto import tqdm
from metrics.train_metrics import binarize_logits, binarize_targets, iou_score, f1_score

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):

    test_loss = 0.0
    test_acc = 0.0
    test_iou = 0.0
    test_f1 = 0.0

    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X = X.to(device)
            y = y.to(device)

            # Ensure y_test has shape [B, 1, H, W] to match logits
            if y.dim() == 3:  # [B,H,W]
                y = y.unsqueeze(1)  # -> [B,1,H,W]

            # Forward: logits [B,1,H,W]
            y_logits = model(X)

            # Loss should use logits directly (BCEWithLogitsLoss)
            loss = loss_fn(y_logits, y.float())
            test_loss += loss.item()

            # our binary masks (int 0/1)
            y_pred = binarize_logits(y_logits, threshold=0.3)  # (B,1,H,W) int
            y_true = binarize_targets(y)                        # (B,1,H,W) int

            # IoU/F1 (your own functions)
            test_iou += iou_score(y_pred, y_true).item()
            test_f1 += f1_score(y_pred, y_true).item()

            # Preds: probabilities -> binary
            y_probs = torch.sigmoid(y_logits)

            # accuracy (per batch)
            accuracy = accuracy_fn(y_probs, y_true)
            test_acc += accuracy.item()


    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    test_iou /= len(data_loader)
    test_f1 /= len(data_loader)


    return {
        "model_name": model.__class__.__name__,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_iou": test_iou,
        "test_f1": test_f1,
    }
