import torch
import torch.nn as nn

class BaseLineCNN(nn.Module): # almost everything in PyTorch inherits from nn.Module
    def __init__(self, in_channels=3, features=[32, 64, 128]):
        # 3 input canals for RGB picture
        # features: List of channels per conv-layer
        super().__init__()


        self.model = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # doubling the canal size to learn miore complex data
        nn.ReLU(),

        nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0), # Output is just one canal (mask: building=1, not building=0)
        )

    def forward(self, x): # passes input trough the model
        return self.model(x)
