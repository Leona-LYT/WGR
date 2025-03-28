import torch
import torch.nn as nn

def regression_net(in_dim=None, out_dim=None, hidden_dims=None, depth=2, width=64, activation='leaky_relu'):
    """
    Creates a flexible regression neural network with configurable architecture.
    
    Args:
        in_dim (int): Input dimension. If None, uses global Xdim if available.
        out_dim (int): Output dimension. If None, uses global Ydim if available.
        hidden_dims (list): List of hidden layer dimensions. If provided, overrides depth and width.
        depth (int): Number of hidden layers if hidden_dims is not provided.
        width (int): Width of first hidden layer if hidden_dims is not provided. 
                     If depth > 1, subsequent layers will have decreasing width.
        activation (str): Activation function: 'leaky_relu', 'relu', 'tanh', or 'sigmoid'.
    
    Returns:
        nn.Sequential: The regression neural network model.
        
    Note:
        In this work, we only provide FNNs for the compared regression models. 
        It is because the CNN architectures used for both the generator function and 
        the regression function are the same.
        For image data, while generative learning incorporates noise into the masked part 
        and uses the entire image as input, the regression method directly fills 
        the masked part with zeros.
    """
    # Handle input and output dimensions
    if in_dim is None:
        if 'Xdim' in globals():
            in_dim = globals()['Xdim']
        else:
            raise ValueError("Input dimension must be provided if Xdim is not defined globally")
    
    if out_dim is None:
        if 'Ydim' in globals():
            out_dim = globals()['Ydim']
        else:
            raise ValueError("Output dimension must be provided if Ydim is not defined globally")
    
    # Configure activation function
    if activation == 'leaky_relu':
        act_fn = lambda: nn.LeakyReLU(0.01, inplace=True)
    elif activation == 'relu':
        act_fn = lambda: nn.ReLU(inplace=True)
    elif activation == 'tanh':
        act_fn = lambda: nn.Tanh()
    elif activation == 'sigmoid':
        act_fn = lambda: nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
    # Configure hidden dimensions
    if hidden_dims is None:
        if depth == 1:
            hidden_dims = [width]
        else:
            # Create decreasing width for deeper networks
            factor = 0.5  # Each layer is half the width of the previous
            hidden_dims = [max(int(width * (factor ** i)), 1) for i in range(depth)]
    
    # Build the network
    layers = []
    prev_dim = in_dim
    
    # Add hidden layers
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(act_fn())
        prev_dim = h_dim
    
    # Add output layer
    layers.append(nn.Linear(prev_dim, out_dim))
    
    return nn.Sequential(*layers)

# Usage examples:
# model = regression_net(in_dim=10, out_dim=1, hidden_dims=[128, 64, 32])
