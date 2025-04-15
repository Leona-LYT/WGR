import torch
import torch.nn as nn

# =============================================================================
# blocks for network architecture
# =============================================================================
class Unflatten(nn.Module):
    def __init__(self, batch_size=None, *dims):
        super(Unflatten, self).__init__()
        self.batch_size = batch_size
        self.dims = dims
        
    def forward(self, x):
        if self.batch_size is None:
            # Use input's batch size
            batch_size = x.size(0)
            return x.view(batch_size, *self.dims)
        else:
            # Use specified batch size
            return x.view(self.batch_size, *self.dims)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UNetBlock(nn.Module):
    def __init__(self, in_size, out_size, is_down=True, normalize=True, dropout=0.0):
        super(UNetBlock, self).__init__()
        
        layers = []
        if is_down:
            # Downsampling block (encoder)
            layers.append(nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False))
            if normalize:
                layers.append(nn.BatchNorm2d(out_size, 0.8))
            layers.append(nn.LeakyReLU(0.2))
        else:
            # Upsampling block (decoder)
            layers.append(nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False))
            if normalize:
                layers.append(nn.BatchNorm2d(out_size, 0.8))
            layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)
        self.is_down = is_down
        
    def forward(self, x, skip_input=None):
        if self.is_down or skip_input is None:
            return self.model(x)
        else:
            x = self.model(x)
            return torch.cat((x, skip_input), 1)
