import os
import pandas as pd
import torch
from tqdm.auto import tqdm

# Initilize GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Randomize for reproducability
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED) # runs on GPU

# Directory for saving the results
os.makedirs("results", exist_ok=True)

CITIES = [
    "amsterdam", "barcelona", "berlin", "brisbane",
    "cairo", "darmstadt", "lisbon", "madrid",
    "melbourne", "mexico_city", "porto"
]

from pipelines.patch_extraction import process_city
for city in tqdm(CITIES):
    process_city(city, patch_size=32)

from pipelines.patch_filtering import patch_filtering
img_patches, mask_patches = patch_filtering(patches_dir="data_patches", threshold=0.6)

from pipelines.data_prep import data_prep
train_loader, val_loader, test_loader = data_prep(img_patches, mask_patches, batch_size=16, train_split=0.7, val_split=0.15, train_transform=None)

from pipelines.training import training
from utils.plot import plot_train_val_loss, plot_metric_curves, visual_overlays
from pipelines.testing import testing

all_results = []
epochs = 150
lr = 0.0008

# U-Net 
from models.unet_model import UNet
model_name = "UNet"
model = UNet(in_channels=4, out_channels=1).to(device)
history = training(train_loader, val_loader, model=model, epochs=epochs, learning_rate=lr, device=device)

plot_train_val_loss(history, path=f"results/{model_name}_loss.png")
plot_metric_curves(history, "val_iou", path=f"results/{model_name}_val_iou.png")
plot_metric_curves(history, "val_f1", path=f"results/{model_name}_val_f1.png")

model_results = testing(model, model_name=model_name, test_loader=test_loader, device=device)
all_results.append(model_results)

visual_overlays(
    model=model,
    data_loader=test_loader,
    device=device,
    save_dir=f"results/overlays_{model_name}",
    num_samples=10,
    threshold=0.5,
    rgb_indices=(2, 1, 0)
)

# Comparison table
df = pd.DataFrame(all_results).round(4)

# optional: sort best-first by IoU or F1
df = df.sort_values(by="test_iou", ascending=False)

print(df.to_markdown(index=False))
df.to_csv("results/final.csv", index=False)

with open("results/final.md", "w") as f:
    f.write(df.to_markdown(index=False))

print("\nSAVED: Please check 'results' dirctory for the model results.")

