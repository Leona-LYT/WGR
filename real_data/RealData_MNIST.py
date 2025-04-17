import sys
import os
current_dir = os.getcwd()  #use to import the defined functions
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)  

"""
incase the above code does not work, you can use the absolute path instead
sys.path.append(r".\")
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from data.image_data import DatasetLoader
from utils.basic_utils import setup_seed, sample_noise
from utils.training_utils import train_WGR_image
from models.generator import generator_fnn
from models.discriminator import discriminator_fnn
from utils.plot_utils import visualize_mnist_digits, visualize_digits, convert_generated_to_mnist_range 

import argparse

if 'ipykernel_launcher.py' in sys.argv[0]:  #if not work in jupyter, you can delete this part
    import sys
    sys.argv = [sys.argv[0]] 


parser = argparse.ArgumentParser(description='Implementation of WGR for MNIST dataset')

parser.add_argument('--Xdim', default=784, type=int, help='dimensionality of X')
parser.add_argument('--Ydim', default=144, type=int, help='dimensionality of Y')

parser.add_argument('--noise_dim', default=100, type=int, help='dimensionality of noise vector')
parser.add_argument('--noise_dist', default='gaussian', type=str, help='distribution of noise vector')

parser.add_argument('--train', default=20000, type=int, help='size of train dataset')
parser.add_argument('--val', default=1000, type=int, help='size of validation dataset')
parser.add_argument('--test', default=10000, type=int, help='size of test dataset')

parser.add_argument('--train_batch', default=100, type=int, metavar='BS', help='batch size while training')
parser.add_argument('--val_batch', default=100, type=int, metavar='BS', help='batch size while validation')
parser.add_argument('--test_batch', default=100, type=int, metavar='BS', help='batch size while testing')

parser.add_argument('--device', default='cpu', type=str, help='device')
args = parser.parse_args()

print(args)

device = 'cpu'
# Initialize dataset loader
loader = DatasetLoader()

# Load subset of MNIST dataset
train_loader, val_loader, test_loader, classes = loader.load_mnist(
    train_size=args.train, val_size=args.val, test_size=args.test, 
    train_batch=args.train_batch, val_batch=args.val_batch, test_batch=args.test_batch,
    mask_size=12, mask_position=(7, 7)
)

# take examples for visualization
eg_examples = enumerate(test_loader) 
batch_idx, (eg_x, eg_y, eg_label) = next(eg_examples) # take samples from the first batch
recon_x = eg_x.clone() # the one used in the training process for visulation
global eg_x, eg_y,  recon_x

# sample 2 samples from each class
num_classes = eg_label.max().item() + 1   
num_samples_per_class = 2

selected_indices = []

for c in range(num_classes):
    class_indices = (eg_label == c).nonzero(as_tuple=True)[0]
    selected = class_indices[:num_samples_per_class]  
    selected_indices.append(selected)
    
selected_indices = torch.cat(selected_indices)
selected_labels = eg_label[selected_indices]

print("Selected indices:", selected_indices)
print("Selected labels:", selected_labels)

recon_x[selected_indices, :, 7:19, 7:19] = eg_y[selected_indices]

#plot the X, Y, and XY
visualize_digits( images=eg_x[selected_indices] , labels = eg_label[selected_indices], figsize=(3, 13), title='X')
visualize_digits( images=eg_y[selected_indices] , labels = eg_label[selected_indices], figsize=(3, 13), title='Y')
visualize_digits( images=recon_x[selected_indices] , labels = eg_label[selected_indices], figsize=(3, 13), title='(X,Y)')

        
setup_seed(5678) 

G = generator_fnn(Xdim=args.Xdim, Ydim=args.Ydim, noise_dim=100, network_type = 'mnist', hidden_dims = [1024, 1024, 512], final_activation = 'tanh')
D = discriminator_fnn(input_dim=784, network_type = 'mnist',  hidden_dims = [256, 256 ])

D_solver = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5,0.999))
G_solver = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5,0.999))  

train_WGR_image(D,G, D_solver,G_solver,Xdim=args.Xdim,Ydim=args.Ydim, noise_dim=args.noise_dim, 
                loader_data=train_loader, loader_val=val_loader, batch_size = args.train_batch, 
                eg_x =eg_x, eg_label=eg_label, selected_indices=selected_indices, lambda_w=0.9, lambda_l=0.1, num_epochs=400)   

eg_eta =  sample_noise(args.train_batch, dim=args.noise_dim ).to(device)
g_exam_input = torch.cat([eg_x.view(args.train_batch, args.Xdim), eg_eta], dim=1)
recon_y = G_net(g_exam_input).view(args.train_batch,1,12,12)
recover_y = convert_generated_to_mnist_range(recon_y)
                        
recon_x = eg_x.clone()
recon_x[selected_indices,:,7:19,7:19] = recover_y[selected_indices,:,:,:].detach()
visualize_digits( images=recon_x[selected_indices] , labels = eg_label[selected_indices], figsize=(3, 13), title='(X,hat(Y)')
