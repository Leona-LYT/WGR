import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions.multivariate_normal import MultivariateNormal

# =============================================================================
# Define data generator for differe models
# =============================================================================

# Independent Gussian Mixture Model

def generate_Gussian_mixture_data(X=None, n_samples=10000):
    """
    Generate data with three clusters based on uniform interval partitioning.
    
    Parameters:
    X (torch.Tensor, optional): Default X input. If None, randomly generated
    n_samples (int): Number of samples to generate
    
    Returns:
    tuple: (X, Y) where X is the input and Y contains the 2D coordinates
    """
    
    # Generate one-dimensional X if not provided
    if X is None:
        X = torch.from_numpy(np.random.normal(0, 1, n_samples))
    else:
        n_samples = len(X)
        
    # Generate uniform samples
    U1 = torch.from_numpy(np.random.uniform(0, 1, n_samples))
    U2 = torch.from_numpy(np.random.uniform(0, 1, n_samples))
    
    # Create interval indicators using tensor operations
    I1U1 = (U1 < 1/3).float()
    I2U1 = ((U1 >= 1/3) & (U1 < 2/3)).float()
    I3U1 = (U1 >= 2/3).float()
    
    I1U2 = (U2 < 1/3).float()
    I2U2 = ((U2 >= 1/3) & (U2 < 2/3)).float()
    I3U2 = (U2 >= 2/3).float()
    
    # Generate cluster-specific noise
    eps1 = (I1U1 * torch.normal(-2, 0.25, (n_samples,)) + 
            I2U1 * torch.normal(0, 0.25, (n_samples,)) + 
            I3U1 * torch.normal(2, 0.25, (n_samples,)))
    
    eps2 = (I1U2 * torch.normal(-2, 0.25, (n_samples,)) + 
            I2U2 * torch.normal(0, 0.25, (n_samples,)) + 
            I3U2 * torch.normal(2, 0.25, (n_samples,)))
    
    # Create final Y values
    Y1 = X + eps1
    Y2 = X + eps2
    Y = torch.stack([Y1, Y2], dim=1)
    
    return X, Y

# Involute Model

def generate_involute_data(X=None, n_samples=10000):
    """
    Generate data forming an involute curve pattern.
    
    Parameters:
    X (torch.Tensor, optional): Default X input. If None, randomly generated
    n_samples (int): Number of samples to generate
    
    Returns:
    tuple: (X, Y) where X is the input and Y contains the 2D coordinates
    """
    # Generate one-dimensional X if not provided
    if X is None:
        X = torch.from_numpy(np.random.normal(0, 1, n_samples))
    else:
        n_samples = len(X)
   
    
    # Generate angle parameter
    U = torch.from_numpy(np.random.uniform(0, 2*math.pi, n_samples))
    
    # Generate noise
    eps1 = torch.normal(0, 0.4, (n_samples,))
    eps2 = torch.normal(0, 0.4, (n_samples,))
    
    # Create final Y values using the involute curve formula
    Y1 = X + U * torch.sin(2*U) + eps1
    Y2 = X + U * torch.cos(2*U) + eps2
    Y = torch.stack([Y1, Y2], dim=1)
    
    return X, Y

# Octagon Gaussian Model


# Octagon Gaussian Model
def generate_octagon_data(X=None, n_samples=10000):
    """
    Generate data points forming an octagon pattern.
    
    Parameters:
    X (torch.Tensor, optional): Default X input. If None, randomly generated
    n_samples (int): Number of samples to generate
    
    Returns:
    tuple: (X, Y) where X is the input and Y contains the 2D coordinates
    """
    # Generate one-dimensional X if not provided
    if X is None:
        X = torch.from_numpy(np.random.normal(0, 1, n_samples))
    else:
        n_samples = len(X)
    
    # Create interval indicator function
    def create_interval_indicators(u_values, n_intervals=8):
        """Create indicators for which interval each u value falls into"""
        # Convert to tensor for easier operations
        u_tensor = u_values.clone()
        # Initialize result tensor
        indicators = torch.zeros((n_samples, n_intervals))
        
        # Set indicators for each interval
        for i in range(n_intervals):
            indicators[:, i] = ((u_tensor > i) & (u_tensor < (i+1))).float()
            
        return indicators
    
    # Generate uniform samples using PyTorch instead of NumPy
    U1 = torch.rand(n_samples) * 8
    U2 = torch.rand(n_samples) * 8
    
    # Get indicators for which interval each sample falls into
    U1_ind = create_interval_indicators(U1)
    U2_ind = create_interval_indicators(U2)
    
    # Calculate means for each octagon vertex
    angles = torch.arange(1, 9).float() * (math.pi / 4)
    mu1 = 3 * torch.cos(angles)
    mu2 = 3 * torch.sin(angles)
    Mu = torch.stack([mu1, mu2], dim=1)
    
    # Calculate covariance matrices for each vertex using PyTorch
    s11 = torch.cos(angles)**2 + 0.16**2 * torch.sin(angles)**2
    s12 = s21 = (1 - 0.16**2) * torch.cos(angles) * torch.sin(angles)
    s22 = torch.sin(angles)**2 + 0.16**2 * torch.cos(angles)**2
    
    # Initialize terms for the sum
    eps1_term = torch.zeros(n_samples, 8)
    eps2_term = torch.zeros(n_samples, 8)
    
    # Generate multivariate normal samples for each vertex using PyTorch
    for i in range(8):
        # Create covariance matrix
        cov_matrix = torch.tensor([[s11[i], s12[i]], 
                                   [s21[i], s22[i]]])
        
        # Use PyTorch's multivariate normal distribution
        mvn = torch.distributions.MultivariateNormal(
            Mu[i], covariance_matrix=cov_matrix
        )
        
        # Sample from the distribution
        multi_norm_i = mvn.sample((n_samples,))
        
        # Multiply by indicators
        eps1_term[:, i] = U1_ind[:, i] * multi_norm_i[:, 0]
        eps2_term[:, i] = U2_ind[:, i] * multi_norm_i[:, 1]
    
    # Sum contributions from each vertex
    eps1 = eps1_term.sum(dim=1)
    eps2 = eps2_term.sum(dim=1)
    
    # Create final Y values
    Y1 = X + eps1
    Y2 = X + eps2
    Y = torch.stack([Y1, Y2], dim=1)
    
    return X, Y

# =============================================================================
# Visualization 
# =============================================================================
def plot_kde_2d(data, cmap='Blues', fill=True, show_cbar=True, thresh=0.05, 
               bw_adjust=0.2, levels=6, fig_size=(8, 6), title=None):
    """
    Create a 2D kernel density estimation plot for bivariate data.
    
    Parameters:
    data (array-like): Input data with shape (n_samples, 2)
    cmap (str): Colormap for the density plot
    fill (bool): Whether to fill the contour
    show_cbar (bool): Whether to show the color bar
    thresh (float): Threshold for the contour plot
    bw_adjust (float): Bandwidth adjustment factor
    levels (int): Number of contour levels
    fig_size (tuple): Figure size (width, height) in inches
    title (str): Optional title for the plot
    
    Returns:
    fig, ax: The matplotlib figure and axis objects
    """
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create the KDE plot
    sns.kdeplot(
        x=data[:, 0], 
        y=data[:, 1],
        cmap=cmap,
        fill=fill,
        cbar=show_cbar,
        thresh=thresh, 
        bw_adjust=bw_adjust, 
        levels=levels,
        common_grid=True,
        cbar_kws={'format': '%.3f'},
        ax=ax
    )
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Return the figure and axis objects for further customization if needed
    return fig, ax
                   
# =============================================================================
# Plots
# =============================================================================
eg_X = torch.from_numpy(np.random.normal(0, 1, 1)) # randomly select a sample X
n_samples = 5000

# Create a tensor with repeated X values
X_repeated = eg_X.repeat(n_samples)

# Generate data in batches
_, eg_Y_GaussMix = generate_Gussian_mixture_data(X=X_repeated)
_, eg_Y_involute = generate_involute_data(X=X_repeated)
_, eg_Y_octagon = generate_octagon_data(X=X_repeated)

# At this point, each Y tensor should have shape [n_samples, 2]
fig, ax = plot_kde_2d(eg_Y_GaussMix, title="Gussian Mixture Model")
plt.show()
plt.close()

fig, ax = plot_kde_2d(eg_Y_involute, title="Involute Model")
plt.show()
plt.close()

fig, ax = plot_kde_2d(eg_Y_octagon, title="Octagon Model")
plt.show()
plt.close()

