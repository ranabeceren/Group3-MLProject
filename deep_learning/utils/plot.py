import matplotlib.pyplot as plt

def plot_train_val_loss(history, path=None):
    """
    Plots training and validation loss over epochs.

    Args:
        history (dict): Dictionary containing lists like:
                        history["train_loss"], history["val_loss"].
        path (str, optional): If given, saves the plot to this path.
    """
    # 1) Create the x-axis values: 1, 2, ..., N (one number per epoch)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 2) Make a new figure (a blank canvas for the plot)
    plt.figure()

    # 3) Plot training loss as a line
    plt.plot(epochs, history["train_loss"], label="Train Loss")

    # 4) Plot validation loss as a second line
    plt.plot(epochs, history["val_loss"], label="Validation Loss")

    # 5) Label the axes
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 6) Add a legend (shows which line is which)
    plt.legend()

    # 7) Add a title
    plt.title("Training vs Validation Loss")

    # 8) Optional: save to file for your report
    if path is not None:
        plt.savefig(path, dpi=200, bbox_inches="tight")

    # 9) Display the plot
    plt.show()

def plot_metric_curves(history, key, path=None):
    """
    Generic plotter for a metric stored in history (e.g. val_iou, val_f1).
    """
    epochs = range(1, len(history[key]) + 1)
    plt.figure()
    plt.plot(epochs, history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.legend()
    plt.title(f"{key} over epochs")
    if path is not None:
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def _to_numpy(x):
    return x.detach().cpu().numpy()

def _minmax01(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)

def visual_overlays(
    model,
    data_loader,
    device,
    save_dir="results/overlays",
    num_samples=10,
    threshold=0.5,
    rgb_indices=(2, 1, 0),  # change if your channel order differs
):
    """
    Saves images showing:
      1) Input RGB
      2) GT overlay on RGB
      3) Pred overlay on RGB

    rgb_indices: tuple of 3 channel indices to visualize as RGB.
      Example if your channels are [B2,B3,B4,B8], use (2,1,0) = (B4,B3,B2).
    """

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    saved = 0

    with torch.inference_mode():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            if y.ndim == 3:
                y = y.unsqueeze(1)  # [B,1,H,W]

            logits = model(X)
            probs = torch.sigmoid(logits)
            pred = (probs >= threshold).float()

            X_np = _to_numpy(X)       # [B,C,H,W]
            y_np = _to_numpy(y)[:, 0] # [B,H,W]
            p_np = _to_numpy(pred)[:, 0]

            B = X_np.shape[0]
            for i in range(B):
                if saved >= num_samples:
                    return

                img = X_np[i]   # [C,H,W]
                gt = y_np[i]    # [H,W]
                pr = p_np[i]    # [H,W]

                r, g, b = rgb_indices
                rgb = np.stack([img[r], img[g], img[b]], axis=-1)  # [H,W,3]
                rgb = _minmax01(rgb)

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))

                ax[0].imshow(rgb)
                ax[0].set_title("Input (RGB)")
                ax[0].axis("off")

                ax[1].imshow(rgb)
                ax[1].imshow(gt, alpha=0.4)
                ax[1].set_title("GT overlay")
                ax[1].axis("off")

                ax[2].imshow(rgb)
                ax[2].imshow(pr, alpha=0.4)
                ax[2].set_title(f"Pred overlay (thr={threshold})")
                ax[2].axis("off")

                plt.tight_layout()
                out_path = os.path.join(save_dir, f"overlay_{saved:02d}.png")
                plt.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

                saved += 1
