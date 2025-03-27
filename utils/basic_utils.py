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
