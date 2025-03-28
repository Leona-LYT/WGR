import sys
import os
current_dir = os.getcwd() 
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)


import torch
import torch.nn as nn
from utils.basic_utils import Flatten, Unflatten, UNetBlock

# =============================================================================
# feedforward neural networks
# =============================================================================
def generator_fnn(Xdim, Ydim, noise_dim, network_type='univariate', hidden_dims=None, activation='leaky_relu', final_activation=None):
    """
    A flexible generator model that can be configured for different use cases.
    All generators take Xdim+noise_dim as input and produce Ydim as output.
    Default FNNs are provided, but you can define your own network as needed.
    
    Args:
        Xdim (int): Dimension of feature input
        Ydim (int): Dimension of output
        noise_dim (int): Dimension of noise input
        network_type (str): 'univariate', 'multivariate', or 'mnist'
        hidden_dims (list): List of hidden layer dimensions. If None, uses defaults based on network_type.
        activation (str): Activation function to use between hidden layers ('leaky_relu' or 'relu').
        final_activation (str or None): Final activation function (None, 'tanh', etc.)
    
    Returns:
        nn.Sequential: The generator model
    """
    input_dim = Xdim + noise_dim
    output_dim = Ydim
    
    # Set default hidden dimensions based on network type
    if hidden_dims is None:
        if network_type == 'univariate':
            hidden_dims = [64, 32]
        elif network_type == 'multivariate':
            hidden_dims = [512, 512, 512]
        elif network_type == 'mnist':
            hidden_dims = [1024, 1024, 512]
        else:
            raise ValueError(f"Unknown network_type: {network_type}")
    
    # Set activation function
    if activation == 'leaky_relu':
        act_layer = lambda: nn.LeakyReLU(0.01, inplace=True)
    else:  # default to relu
        act_layer = lambda: nn.ReLU(inplace=True)
    
    # Set final activation for MNIST if not specified
    if final_activation is None and network_type == 'mnist':
        final_activation = 'tanh'
    
    # Build the network
    layers = []
    current_dim = input_dim
    
    # Add hidden layers with activation
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(act_layer())
        current_dim = h_dim
    
    # Add output layer
    layers.append(nn.Linear(current_dim, output_dim))
    
    # Add final activation if specified
    if final_activation == 'tanh':
        layers.append(nn.Tanh())
    elif final_activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    
    return nn.Sequential(*layers)

# =============================================================================
# Convolutional neural networks
# =============================================================================
def generator_cnn(generator_type='dcgan', input_shape=None, Xdim=None, noise_dim=None, batch_size=None):
    """
    Unified generator function that creates different types of generators.
    
    Args:
        generator_type (str): 'dcgan' or 'unet'
        input_shape (tuple): Input shape (C, H, W) for UNet generator
        Xdim (int): Dimension of feature input for DCGAN
        noise_dim (int): Dimension of noise input for DCGAN
        batch_size (int): Batch size for Unflatten operation
        
    Returns:
        nn.Module: The generator model
    """
    if generator_type.lower() == 'dcgan':
        # Set defaults for noise_dim
        if noise_dim is None:
            if 'args' in globals() and hasattr(args, 'noise'):
                noise_dim = args.noise
            else:
                noise_dim = 100
        
        # Set defaults for Xdim
        if Xdim is None:
            if 'Xdim' in globals():
                Xdim = globals()['Xdim']
            else:
                Xdim = 784  # Default for MNIST-like data
        
        # Total input dimension is Xdim + noise_dim
        input_dim = Xdim + noise_dim
                
        if batch_size is None and 'args' in globals() and hasattr(args, 'batch'):
            batch_size = args.batch
            
        return nn.Sequential(
            # Linear layers with ReLU and BatchNorm
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 3*3*128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(3*3*128),
            
            # Reshape to convolutional format
            Unflatten(batch_size, 128, 3, 3),
            
            # Transposed convolutions
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
        
    elif generator_type.lower() == 'unet':
        # Handle UNet generator
        if input_shape is None:
            raise ValueError("input_shape is required for UNet generator")
            
        class UNetGenerator(nn.Module):
            def __init__(self, input_channels):
                super(UNetGenerator, self).__init__()
                
                # Encoder (downsampling)
                self.down1 = UNetBlock(input_channels, 64, is_down=True, normalize=False)
                self.down2 = UNetBlock(64, 128, is_down=True)
                self.down3 = UNetBlock(128, 256, is_down=True, dropout=0.5)
                self.down4 = UNetBlock(256, 512, is_down=True, dropout=0.5)
                self.down5 = UNetBlock(512, 512, is_down=True, dropout=0.5)
                self.down6 = UNetBlock(512, 512, is_down=True, dropout=0.5)
                
                # Decoder (upsampling)
                self.up1 = UNetBlock(512, 512, is_down=False, dropout=0.5)
                self.up2 = UNetBlock(1024, 512, is_down=False, dropout=0.5)
                self.up3 = UNetBlock(1024, 256, is_down=False, dropout=0.5)
                self.up4 = UNetBlock(512, 128, is_down=False)
                self.up5 = UNetBlock(256, 64, is_down=False)
                
                # Final layer
                self.final = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(128, input_channels, 3, 2, 2),
                    nn.Tanh()
                )
                
            def forward(self, x):
                # Encoder
                d1 = self.down1(x)
                d2 = self.down2(d1)
                d3 = self.down3(d2)
                d4 = self.down4(d3)
                d5 = self.down5(d4)
                d6 = self.down6(d5)
                
                # Decoder with skip connections
                u1 = self.up1(d6, d5)
                u2 = self.up2(u1, d4)
                u3 = self.up3(u2, d3)
                u4 = self.up4(u3, d2)
                u5 = self.up5(u4, d1)
                
                return self.final(u5)
                
        return UNetGenerator(input_shape[0])
        
    else:
        raise ValueError(f"Unknown generator_type: {generator_type}")


# Usage examples:
# gen_univariate = generator_fnn(Xdim=10, Ydim=1, noise_dim=100, network_type='univariate')
# gen_multivariate = generator_fnn(Xdim=10, Ydim=5, noise_dim=100, network_type='multivariate')
# gen_mnist = generator_fnn(Xdim=784, Ydim=144, noise_dim=100, network_type='mnist')
# gen_custom = generator_fnn(Xdim=784, Ydim=144,  noise_dim=100, hidden_dims=[256, 128, 64], activation='relu', final_activation='tanh')
# dcgan_gan = generator_cnn('dcgan', Xdim=10, noise_dim=100, batch_size=64)
# unet_gen = generator_cnn('unet', input_shape=(3, 128, 128))
