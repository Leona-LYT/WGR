import torch
import torch.nn as nn
import random
import numpy as np

#Random seeds
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)

def reset_grad():
    D_solver.zero_grad()
    G_solver.zero_grad()

# Lipschitz Continuous Constraint
def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP"""
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
def discriminator_loss(logits_real, logits_fake):

    real_image_loss = torch.mean(logits_real)
    fake_image_loss = torch.mean(logits_fake)

    loss = real_image_loss - fake_image_loss  

    return loss

def generator_loss(logits_fake):

    loss = torch.mean(logits_fake)

    return loss

l1_loss = nn.L1Loss()  # loss(input,target)
l2_loss = nn.MSELoss()

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
