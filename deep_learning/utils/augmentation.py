import torch

class DataAugmentations:
    """
    Two augmentations:
      - random flips (H/V)
      - random rot90
    Applies same transform to image and mask.
    """

    def __init__(self, flip=0.5, rotate=0.5):
        self.flip = flip
        self.rotate = rotate

    def __call__(self, X, y):
        # X: (B,C,H,W), y: (B,1,H,W) or (B,H,W)
        if y.ndim == 3:
            y = y.unsqueeze(1)

        B = X.shape[0]
        X_out, y_out = [], []

        for i in range(B):
            img = X[i]
            mask = y[i]

            # flips
            if torch.rand(1).item() < self.flip:
                if torch.rand(1).item() < 0.5:  # horizontal
                    img = torch.flip(img, dims=[2])
                    mask = torch.flip(mask, dims=[2])
                else:  # vertical
                    img = torch.flip(img, dims=[1])
                    mask = torch.flip(mask, dims=[1])

            # rotates
            if torch.rand(1).item() < self.rotate:
                k = int(torch.randint(0, 4, (1,)).item())
                img = torch.rot90(img, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[1, 2])

            X_out.append(img)
            y_out.append(mask)

        return torch.stack(X_out, dim=0), torch.stack(y_out, dim=0)