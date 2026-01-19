import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module): # feature extraction without changing the size
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super().__init__()

        self.double_conv = nn.Sequential( # runs the layers one after the other
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # number of channels goes up
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )

    def forward(self, x): # runs through the two Conv-Layers
        return self.double_conv(x)


class Down(nn.Module): # Encoder
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential( #downsampling (height and width in half)
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels) #again feature extraction
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module): # Decoder
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2 # doubles height and weight again
        )

        self.conv = DoubleConv(in_channels, out_channels) #combining features

    def forward(self, x1, x2): #x1 from decoder, x2 from encoder
        x1 = self.up(x1) #Upscaling the decoded features
        x = torch.cat([x2, x1], dim=1) # Skip connection
        x = self.conv(x)

        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) #mixes the channels (is this pixel a building or not?)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        # Images get smaller, meaning bigger
        self.income = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        #self.down3 = Down(128, 256)

        # Image gets bigger again and includes skip connection
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        #self.up3 = Up(64, 32)

        self.outcome = OutConv(32, out_channels) # final pixel classification

    def forward(self, x): #datastream
        x1 = self.income(x) # very small details
        x2 = self.down1(x1) # then it gets more abstract
        x3 = self.down2(x2)
        #x4 = self.down3(x3)

        x = self.up1(x3, x2) #decoding and skip connection
        x = self.up2(x, x1)
        # x = self.up4(x

        return self.outcome(x)