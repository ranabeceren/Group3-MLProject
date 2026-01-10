import torch
import torch.nn as nn # contains all the NN Modules such as ReLU, Sequential...

class BaseLineCNN(nn.Module): # almost everything in PyTorch inherits from nn.Module
    def __init__(self, in_channels=3, features=[32, 64, 128]):
        # 3 input canals for RGB picture
        # list of the canals for the Conv-Layer
        super().__init__() # Constructor

        # dynamic building of the layers
        layers = []
        prev_channels = in_channels # number of canals for the output of the layer

        for feature in features:
            layers.append(nn.Conv2d(in_channels=prev_channels, out_channels=feature, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
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
