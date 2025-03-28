import torch
import torch.nn as nn
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BNN(PyroModule):
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

# Usage examples:
# B_net = BNN(in_dim=10, out_dim=1, hidden_dims=[32, 16, 8])
