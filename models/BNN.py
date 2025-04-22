import torch
import torch.nn as nn
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class Bayesian_fnn(PyroModule):
    def __init__(
        self, 
        in_dim=100, 
        out_dim=1, 
        hidden_dims=None, 
        depth=1, 
        width=2, 
        activation='tanh', 
        prior_scale=1.,
        likelihood='gaussian'
    ):
        """
        Flexible Bayesian Neural Network with configurable architecture and priors.
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            hidden_dims (list): List of hidden layer dimensions. If provided, overrides depth and width.
            depth (int): Number of hidden layers (used if hidden_dims is None)
            width (int): Width of hidden layers (used if hidden_dims is None)
            activation (str): Activation function: 'tanh', 'relu', 'leaky_relu', 'sigmoid', or 'gelu'
            prior_scale (float): Scale of the prior distribution
            likelihood (str): Type of likelihood: 'gaussian', 'bernoulli', or 'categorical'
        """
        super().__init__()
        
        # Configure activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Configure hidden dimensions based on depth and width if not provided
        if hidden_dims is None:
            hidden_dims = [width] * depth
        
        # Validate dimensions
        assert in_dim > 0 and out_dim > 0 and all(hd > 0 for hd in hidden_dims)
        
        # Store network configuration
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.prior_scale = prior_scale
        self.likelihood = likelihood
        
        # Define the layer sizes including input and output
        self.layer_sizes = [in_dim] + hidden_dims + [out_dim]
        
        # Create PyroModule layers
        layer_list = [
            PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) 
            for idx in range(1, len(self.layer_sizes))
        ]
        self.layers = PyroModule[nn.ModuleList](layer_list)
        
        # Initialize weights and biases with prior distributions
        self._init_priors()
    
    def _init_priors(self):
        """Initialize model parameters with prior distributions."""
        for layer_idx, layer in enumerate(self.layers):
            # Weight priors - scale by sqrt(2/fan_in) for better initialization
            layer.weight = PyroSample(
                dist.Normal(
                    0., 
                    self.prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx])
                ).expand([self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2)
            )
            
            # Bias priors
            layer.bias = PyroSample(
                dist.Normal(0., self.prior_scale).expand([self.layer_sizes[layer_idx + 1]]).to_event(1)
            )
    
    def forward(self, x, y=None):
        """
        Forward pass through the Bayesian neural network.
        
        Args:
            x (torch.Tensor): Input data
            y (torch.Tensor, optional): Observed outputs for Bayesian inference
            
        Returns:
            torch.Tensor: Model predictions
        """
        # Ensure input has the right shape
        if x.dim() == 1:
            x = x.unsqueeze(1)
            
        # Forward pass through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        
        # Output layer (no activation for regression)
        mu = self.layers[-1](x)
        
        # Handle different output dimensions
        if self.out_dim == 1:
            mu = mu.squeeze(-1)
        
        # Model the likelihood based on the specified type
        if y is not None:
            with pyro.plate("data", x.shape[0]):
                if self.likelihood == 'gaussian':
                    # For regression: Normal likelihood with learned sigma
                    sigma = pyro.sample("sigma", dist.Gamma(1.0, 1.0))
                    obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
                elif self.likelihood == 'bernoulli':
                    # For binary classification: Bernoulli likelihood
                    probs = torch.sigmoid(mu)
                    obs = pyro.sample("obs", dist.Bernoulli(probs), obs=y)
                elif self.likelihood == 'categorical':
                    # For multi-class classification: Categorical likelihood
                    logits = mu if self.out_dim > 1 else mu.unsqueeze(-1)
                    obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
                else:
                    raise ValueError(f"Unsupported likelihood: {self.likelihood}")
        
        return mu


class Bayesian_cnn(PyroModule):
    def __init__(self, 
                 input_channels=1, 
                 hidden_channels=(16, 32), 
                 output_dim=144, 
                 input_size=(28, 28),
                 w_scale=1.0,
                 b_scale=10.0):
        """Flexible Bayesian CNN with configurable architecture"""
        super().__init__()
        
        # Store configuration
        self.input_size = input_size
        self.input_channels = input_channels
        
        # Create layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        # Build conv layers dynamically
        for i, out_channels in enumerate(hidden_channels):
            conv = PyroModule[nn.Conv2d](
                in_channels, out_channels, 
                kernel_size=3, 
                stride=1 if i == 0 else 2,
                padding=1
            )
            # Set priors
            conv.weight = PyroSample(dist.Normal(0., w_scale).expand([out_channels, in_channels, 3, 3]).to_event(4))
            conv.bias = PyroSample(dist.Normal(0., b_scale).expand([out_channels]).to_event(1))
            self.conv_layers.append(conv)
            in_channels = out_channels
        
        # Pool and activation
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
        # Calculate flattened size
        h, w = input_size
        h, w = h//2, w//2  # First pooling
        for _ in range(1, len(hidden_channels)):
            h, w = (h+1)//2, (w+1)//2  # Subsequent layers with stride=2
        feature_size = hidden_channels[-1] * h * w
        
        # FC layer
        self.fc = PyroModule[nn.Linear](feature_size, output_dim)
        self.fc.weight = PyroSample(dist.Normal(0., w_scale).expand([output_dim, feature_size]).to_event(2))
        self.fc.bias = PyroSample(dist.Normal(0., b_scale).expand([output_dim]).to_event(1))
    
    def forward(self, x, y=None):
        # Ensure correct input shape
        if x.dim() == 2:
            x = x.view(-1, self.input_channels, *self.input_size)
        
        # Apply conv layers
        for i, conv in enumerate(self.conv_layers):
            x = self.relu(conv(x))
            if i == 0:  # Pool only after first conv
                x = self.pool(x)
        
        # FC layer
        x = self.fc(x.view(x.size(0), -1))
        x = torch.sigmoid(x)
        
        # Likelihood
        if y is not None:
            with pyro.plate("data", x.size(0)):
                pyro.sample("obs", dist.Normal(x, torch.ones_like(x)).to_event(1), obs=y)
        
        return x


def BNN_CP(true_Y, LB, UB, sample_size):
    CP = torch.zeros([sample_size])
    for i in range(sample_size):
        if true_Y[i] > LB[i] and true_Y[i] < UB[i]:
            CP[i] = 1
    print(CP.sum()/sample_size)
    return CP.sum()/sample_size
    
# Usage examples:
# B_net = Bayesian_fnn(in_dim=10, out_dim=1, hidden_dims=[32, 16, 8])
# B_mnist = Bayesian_fnn(input_channels=1, hidden_channels=(32, 64), output_dim=144, input_size=(28, 28))
# B_stl10 = Bayesian_fnn(input_channels=3, hidden_channels=(32, 64, 128, 256), output_dim=3*65*65, input_size=(128, 128))
