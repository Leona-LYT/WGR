import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Various random seeds
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)

# =============================================================================
# noise generation
# =============================================================================
def sample_noise(size, dim):
    
    temp = torch.randn(size, dim)

    return temp

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

