"""
Simulation studies with univaraite Y. (M1, M2, SM1, SM2, SM3)
"""

import sys
import os
current_dir = os.getcwd()  #use to import the defined functions
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)  

"""
incase the above code does not work, you can use the absolute path instead
sys.path.append(r".\")
"""

from utils.basic_utils import setup_seed, sample_noise, calculate_gradient_penalty, discriminator_loss, generator_loss
import data.SimulationData 
from utils.training_utils import train_WGR_fnn

import torch
import torch.nn as nn
import numpy as np

