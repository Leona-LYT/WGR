import torch
import torch.nn as nn
from basic_utils import Flatten, Unflatten
# =============================================================================
# feedforward networks
# =============================================================================
def discriminator_fnn(input_dim=None, output_dim=1, hidden_dims=None, network_type='univariate', flatten=False, leaky_relu_slope=0.01):
    """
    A flexible discriminator model that can be configured for different use cases.
    The FNNs are considered for tabular data and MNIST.
    
    Args:
        input_dim (int): Dimension of input features.
                         If None, will use Xdim + Ydim for both univariate and multivariate responses
        output_dim (int): Dimension of output. Default is 1 for binary classification.
        hidden_dims (list): List of hidden layer dimensions.
                           Default configurations:
                           - Univariate: [64, 32]
                           - Multivariate: [512, 512, 512]
                           - MNIST: [256, 256]
        network_type (str): Type of network to use - 'univariate', 'multivariate', or 'mnist'
        flatten (bool): Whether to flatten the input tensor. Default is False.
        leaky_relu_slope (float): Negative slope for LeakyReLU activation.
    
    Returns:
        nn.Sequential: The discriminator model
    """
    layers = []
    
    # Add flatten layer if specified or if using MNIST type
    if flatten or network_type == 'mnist':
        layers.append(Flatten())
    
    # Configure default settings based on input dimension
    if input_dim is None:
        if 'args' in globals() and hasattr(args, 'Xdim'):
            input_dim = args.Xdim + args.Ydim  # Use args.Xdim + args.Ydim for both univariate and multivariate
        elif network_type == 'mnist':
            input_dim = 784  # Default for MNIST
    
    # Set default hidden dimensions based on network type
    if hidden_dims is None:
        if network_type == 'mnist':
            hidden_dims = [256, 256]
        elif network_type == 'multivariate':
            hidden_dims = [512, 512, 512]
        else:  # univariate
            hidden_dims = [64, 32]
    
    # Build the network
    current_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
        current_dim = h_dim
    
    # Output layer with configurable output dimension
    layers.append(nn.Linear(current_dim, output_dim))
    
    return nn.Sequential(*layers)

# =============================================================================
# convolutional networks
# =============================================================================
def discriminator_cnn(network_type='mnist', batch_size=None, in_channels=None, use_sigmoid=False, input_shape=None):
    """
    A flexible CNN discriminator that supports different architectures, which are used for image analysis.
    
    Args:
        network_type (str): 'mnist', 'stl10', or 'patchgan'
        batch_size (int): Batch size for MNIST unflatten operation. Uses args.batch if None
        in_channels (int): Number of input channels. Default varies by network_type
        use_sigmoid (bool): Whether to use sigmoid activation at the end
        input_shape (tuple): (C, H, W) for PatchGAN discriminator
        
    Returns:
        nn.Sequential or nn.Module: The CNN discriminator model
    """
    if network_type == 'mnist':
        # Set defaults for MNIST
        if batch_size is None and 'args' in globals() and hasattr(args, 'batch'):
            batch_size = args.batch
        if in_channels is None:
            in_channels = 1
            
        layers = [
            Unflatten(batch_size, in_channels, 28, 28),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(4*4*64, 4*4*64),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Linear(4*4*64, 1)
        ]
        
    elif network_type == 'stl10':
        # Set defaults for STL10
        if in_channels is None:
            in_channels = 3
            
        # Helper function to create conv-bn-leakyrelu blocks
        def conv_block(in_ch, out_ch, kernel=4, stride=2, padding=1, bias=False):
            return [
                nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            
        # Build the STL10 architecture using the helper function
        layers = []
        channels = [in_channels, 64*2, 64*2, 64*4, 64*4, 64*8]
        
        for i in range(len(channels)-1):
            layers.extend(conv_block(channels[i], channels[i+1]))
            
        # Add final layers
        layers.extend([
            nn.Conv2d(64*8, 1, 4, 1, 0, bias=False),
            Flatten()
        ])
        
    elif network_type == 'patchgan':
        # For PatchGAN, return a module instead of Sequential
        class PatchGANDiscriminator(nn.Module):
            def __init__(self, input_shape):
                super(PatchGANDiscriminator, self).__init__()
                channels, height, width = input_shape
                # Calculate output shape
                patch_h, patch_w = int(height / 2 ** 3), int(width / 2 ** 3)
                self.output_shape = (1, patch_h, patch_w)
                
                def discriminator_block(in_filters, out_filters, stride, normalize):
                    """Returns layers of each discriminator block"""
                    layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
                    if normalize:
                        layers.append(nn.InstanceNorm2d(out_filters))
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                    return layers
                
                layers = []
                in_filters = channels
                for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
                    layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
                    in_filters = out_filters
                
                layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
                self.model = nn.Sequential(*layers)
                self.linear = nn.Sequential(
                    Flatten(),
                    nn.Linear(16*16, 1)
                )
                
                if use_sigmoid:
                    self.linear.add_module('sigmoid', nn.Sigmoid())
                
            def forward(self, img):
                x = self.model(img)
                x = self.linear(x)
                return x
                
        # Check if input_shape is provided
        if input_shape is None:
            input_shape = (3, 256, 256)  # Default size for many image tasks
            
        return PatchGANDiscriminator(input_shape)
        
    else:
        raise ValueError(f"Unknown network_type: {network_type}")
        
    # Add sigmoid if requested (only for sequential models)
    if use_sigmoid:
        layers.append(nn.Sigmoid())
        
    return nn.Sequential(*layers)

# Usage examples:
#FNN
# uniY_disc =discriminator_fnn(input_dim=10, output_dim=1)
# multiY_disc = discriminator_fnn(input_dim=10, output_dim=10, network_type='multivariate')
# mnist_disc = discriminator_fnn(network_type='mnist', output_dim=1)
#CNN
# mnist_disc = discriminator_cnn(network_type='mnist')
# stl10_disc = discriminator_cnn(network_type='stl10', use_sigmoid=True)
# patchgan_disc = discriminator_cnn(network_type='patchgan', input_shape=(3, 256, 256), use_sigmoid=True)
