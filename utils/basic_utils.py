import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributions as dist

# =============================================================================
#Random seeds
# =============================================================================
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False

def get_dimension(x):
    if len(x.shape) == 1:
        return 1  
    else:
        return x.shape[-1]  

# =============================================================================
# noise generation
# =============================================================================
def sample_noise(size, dim, distribution='gaussian', mu=None, cov=None, a=None, b=None, loc=None, scale=None):
    """
    Generate noise samples from various distributions.
    
    Parameters:
        size (int): Number of samples to generate
        dim (int): Dimension of each noise sample
        distribution (str): Distribution type ('gaussian', 'multivariate_gaussian', 'uniform', 'laplace')
        mu (torch.Tensor): Mean vector for multivariate Gaussian
        cov (torch.Tensor): Covariance matrix for multivariate Gaussian
        a (float): Lower bound for uniform distribution
        b (float): Upper bound for uniform distribution
        loc (torch.Tensor): Location parameter for Laplace distribution
        scale (torch.Tensor): Scale parameter for Laplace distribution
    
    Returns:
        torch.Tensor: Noise samples of shape [size, dim]
    """
    if distribution == 'gaussian':
        # Standard Gaussian
        return torch.randn(size, dim)
    
    elif distribution == 'multivariate_gaussian':
        # Multivariate Gaussian
        if mu is None:
            mu = torch.zeros(dim)
        if cov is None:
            cov = torch.eye(dim)
        mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
        return mvn.sample((size,))
    
    elif distribution == 'uniform':
        # Uniform distribution
        if a is None:
            a = 0.0
        if b is None:
            b = 1.0
        return a + (b - a) * torch.rand(size, dim)
    
    elif distribution == 'laplace':
        # Laplace distribution
        if loc is None:
            loc = torch.zeros(dim)
        if scale is None:
            scale = torch.ones(dim)
        laplace_dist = torch.distributions.Laplace(loc, scale)
        return laplace_dist.sample((size,))
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Supported types are: 'gaussian', 'multivariate_gaussian', 'uniform', 'laplace'")

# =============================================================================
# evaluation on the selection of m
# =============================================================================
def selection_m(G, x, y, noise_dim, Xdim, Ydim, train_size ):
    """
    Calculate selection criterion for m, which is the dimension of noise vector eta.
    
    Args:
        G: generator
        x: Covariates of training data
        y: Predictors of training data
        noise_dim: Dimension of noise vector
        Xdim: Dimension of input features
        Ydim: Dimension of response
        threshold: A threshold to decide which penalization is used.
        train_size: Number of samples

    Returns:
        float: Selection criterion value (lower is better)
    """

    with torch.no_grad():
        
        # compute L2 error
        output = torch.zeros([500,train_size,Ydim])
        for i in range(500):
            eta = sample_noise(size=train_size, dim=noise_dim)
            g_input =  torch.cat([x.view([train_size,Xdim]),eta],dim=1)
            output[i] = G(g_input.float()).detach()
        
        if Ydim == 1:
            L2_value = l2_loss(output.mean(dim=0), y.view([train_size,Ydim]))
        if Ydim > 1:
            L2_value = torch.mean(l2_loss(output.mean(dim=0), y),dim=0)
        # Calculate complexity penalty   
        complexity_penalty = (Xdim + noise_dim) * np.log(train_size) / train_size
        
        # Calculate selection criterion
        value_m = L2_value + complexity_penalty

    return value_m
# =============================================================================
# Lipschitz Continuous Constraint
# =============================================================================
def reset_grad():
    D_solver.zero_grad()
    G_solver.zero_grad()


def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty
    
# =============================================================================
# define the loss functions
# =============================================================================
l1_loss = nn.L1Loss()  
l2_loss = nn.MSELoss()

def discriminator_loss(logits_real, logits_fake):

    real_image_loss = torch.mean(logits_real)
    fake_image_loss = torch.mean(logits_fake)

    loss = real_image_loss - fake_image_loss  

    return loss

def generator_loss(logits_fake):

    loss = torch.mean(logits_fake)

    return loss


