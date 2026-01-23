import torch
import torch.nn as nn

class CNNBaselineModel(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int
    ):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
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

        # x4 = self.conv_block_4(x3)

        # x5 = self.conv_block_5(x4)
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
        
class CNNBaselineModel(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int
    ):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    
        )
        '''
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  
        )
    
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  
        )
        '''
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

        # x4 = self.conv_block_4(x3)

        # x5 = self.conv_block_5(x4)
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
        

class BaseLineCNN(nn.Module): # almost everything in PyTorch inherits from nn.Module
    def __init__(self, in_channels=4, features=[32, 64, 128, 256]):
        # 3 input canals for RGB picture
        # list of the canals for the Conv-Layer
        super().__init__() # Constructor

        # dynamic building of the layers
        layers = []
        prev_channels = in_channels # number of canals for the output of the layer

        for feature in features:
            layers.append(nn.Conv2d(in_channels=prev_channels, out_channels=feature, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(feature)) #for more stable training
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.ReLU(inplace=True))

            prev_channels = feature # Input for the next layer

        # last layer returns just 1 canal (for the mask)
        layers.append(nn.Conv2d(in_channels=prev_channels, out_channels=1, kernel_size=1, padding=0))

        '''
        The code above is equivalent to this below, just the dynamic version so changes can be done 
        without having to write everything manually:
        
        
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
        nn.ReLU(),

        nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0), 
        '''

        self.model = nn.Sequential(*layers) # connects all layers to a pipeline and gives the list as single arguments

    def forward(self, x, debug=False):
        # x = Input-tensor [B, C, H, W]
        # debug continues to print the shapes after each layer if True

        if debug:
            out = x
            for layer in self.model: # loop through all layers
                out = layer(out) # puts the input tensor in each layer
                print(f"{layer.__class__.__name__}: {out.shape}") # shows us the layer-type and output shape
            return out
        else:
            return self.model(x) # normal mode: runs input through all layers

# device agnostic code, if GPU is available use GPU, if not then CPU (model works on any computer)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
