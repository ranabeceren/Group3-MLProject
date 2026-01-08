import torch
import torch.nn as nn

class BaseLineCNNModelV3(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int
    ):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
                ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)    
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)    
        )

        self.upsample = nn.Upsample(
            scale_factor=8,   # 32 → 128 (you upscale for the data to fit in the bcewithlogits loss imput shape)
            mode="bilinear",
            align_corners=False
        )

        self.classifier = nn.Conv2d(
            in_channels=hidden_units,
            out_channels=output_shape, 
            kernel_size=1
            )
        
    def forward(self, x):
        input_size = x.shape[-2:]
        x1 = self.conv_block_1(x)
        # print(f"Output shape of conv_block_1: {x.shape}")

        x2 = self.conv_block_2(x1)
        # print(f"Output shape of conv_block_2: {x.shape}")

        x3 = self.conv_block_3(x2)
        """
        x = self.upsample(x3)
        # print(f"Output shape of unsample: {x.shape}")
        """
        x = torch.nn.functional.interpolate(
            x3,
            size=x1.shape[-2:],   # match x1 exactly
            mode="bilinear",
            align_corners=False
            )

        x = x + x1 # to skip connection

        x = torch.nn.functional.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False
            )


        x = self.classifier(x)
        # print(f"Output shape of classifier: {x.shape}")

        return x
