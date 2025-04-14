"""
This is the code for real data analysis with one dimensional response
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

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.model_selection import train_test_split

from utils.basic_utils import setup_seed, get_dimension
from models.generator import generator_fnn
from models.discriminator import discriminator_fnn
from utils.training_utils import train_WGR_fnn
from utils.evaluation_utils import eva_G_UniY

import argparse

if 'ipykernel_launcher.py' in sys.argv[0]:  #if not work in jupyter, you can delete this part
    import sys
    sys.argv = [sys.argv[0]] 


parser = argparse.ArgumentParser(description='Implementation of WGR for CT slices dataset')

parser.add_argument('--Xdim', default=100, type=int, help='dimensionality of X')
parser.add_argument('--Ydim', default=1, type=int, help='dimensionality of Y')

parser.add_argument('--noise_dim', default=30, type=int, help='dimensionality of noise vector')
parser.add_argument('--noise_dist', default='gaussian', type=str, help='distribution of noise vector')

parser.add_argument('--train', default=40000, type=int, help='size of train dataset')
parser.add_argument('--val', default=3500, type=int, help='size of validation dataset')
parser.add_argument('--test', default=10000, type=int, help='size of test dataset')

parser.add_argument('--train_batch', default=128, type=int, metavar='BS', help='batch size while training')
parser.add_argument('--val_batch', default=100, type=int, metavar='BS', help='batch size while validation')
parser.add_argument('--test_batch', default=100, type=int, metavar='BS', help='batch size while testing')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train')

args = parser.parse_args()

print(args)

# import data
all_CT = pd.read_csv("../data/CT.csv") # you can change it to other dataset
all_CT = all_CT.iloc[:, 1:]  # if the first column need to be omitted


#split data into training dataset, testing dataset and validation dataset
train_val_data, test_data = train_test_split(all_CT, test_size=args.test, random_state=5678)
train_data, val_data = train_test_split(train_val_data, test_size=args.val, random_state=5678)

# Convert pandas DataFrames to PyTorch tensors
X_train = torch.tensor(train_data.values[:, :-1], dtype=torch.float32)
y_train = torch.tensor(train_data.values[:, -1], dtype=torch.float32)

X_val = torch.tensor(val_data.values[:, :-1], dtype=torch.float32)
y_val = torch.tensor(val_data.values[:, -1], dtype=torch.float32)

X_test = torch.tensor(test_data.values[:, :-1], dtype=torch.float32)
y_test = torch.tensor(test_data.values[:, -1], dtype=torch.float32)

args.Xdim = get_dimension(X_train)
args.Ydim = get_dimension(y_train)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
loader_train = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False)
loader_test = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

setup_seed(5678)  
# Define generator network and discriminator network
G_net = generator_fnn(Xdim=args.Xdim, Ydim=args.Ydim, noise_dim=args.noise_dim, hidden_dims = [64, 32])
D_net = discriminator_fnn(input_dim=args.Xdim+args.Ydim, hidden_dims = [64, 32])

# Initialize RMSprop optimizers
D_solver = optim.Adam(D_net.parameters(), lr=0.001, betas=(0.9, 0.999))
G_solver = optim.Adam(G_net.parameters(), lr=0.001, betas=(0.9, 0.999))

# Train the model using the train_model function
trained_G, trained_D = train_WGR_fnn(D=D_net, G=G_net, D_solver=D_solver, G_solver=G_solver, loader_train = loader_train, 
                                     loader_val=loader_val, noise_dim=args.noise_dim, Xdim=args.Xdim, Ydim=args.Ydim, 
                                     lambda_w=0.9, lambda_l=0.1, batch_size=args.train_batch, save_path='./',
                                     model_type='CT', device='cpu', num_epochs=200)

CT_numerical_results = eva_G_UniY(G=trained_G, loader_data=loader_test, Ydim=args.Ydim, noise_dim=args.noise_dim, batch_size=args.test_batch, J_t_size=50)
