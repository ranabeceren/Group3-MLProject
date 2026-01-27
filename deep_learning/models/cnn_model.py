import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Baseline(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int ):

        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 32 -> 16
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 16 -> 8
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 8 -> 4
        )
        self.classifier = nn.Conv2d(in_channels=hidden_units, out_channels=output_shape, kernel_size=1)
        
    def forward(self, x):
        
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = self.classifier(x)

        # Upsample it back to patch size
        x = F.interpolate(x, size=(32,32), mode="bilinear", align_corners=False)
        return x
        
class CNN_SkipConnection(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):

        super().__init__()

        c1 = hidden_units # 32
        c2 = c1 * 2       # 64
        c3 = c2 * 2       # 128

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 32 -> 16
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 16 -> 8
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c3, out_channels=c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c3, out_channels=c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 8 -> 4
        )
        self.skip_proj = nn.Conv2d(c1, c3, kernel_size=1)

        self.classifier = nn.Conv2d(in_channels=c3, out_channels=output_shape, kernel_size=1)
        
    def forward(self, x):
        
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)

        # 1) Upsample deep features to match x1 spatial size (4x4 -> 16x16)
        x3_up = F.interpolate(x3, size=x1.shape[-2:], mode="bilinear", align_corners=False)

        # 2) Project skip so channels match (32 -> 128)
        x1_proj = self.skip_proj(x1)  # (B, 128, 16, 16)

        # 3) Fuse (skip connection)
        x_fused = x3_up + x1_proj
        x_fused = F.relu(x_fused)     

        # 4) Predict logits at 16x16, then upsample logits to input size
        logits = self.classifier(x_fused)  # (B, 1, 16, 16)
        logits = F.interpolate(logits, size=(32,32), mode="bilinear", align_corners=False)

        return logits

