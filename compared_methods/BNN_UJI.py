"""
This is the code for Bayesian Neural Network method in the real data analysis with multi-dimensional response.
"""
import sys
import os
current_dir = os.getcwd()  #use to import the defined functions
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)  

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pyro
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive

from models.BNN import Bayesian_fnn, BNN_CP
from utils.basic_utils import setup_seed, get_dimension, bnn_evaluation

import argparse

if 'ipykernel_launcher.py' in sys.argv[0]:  #if not work in jupyter, you can delete this part
    import sys
    sys.argv = [sys.argv[0]] 


parser = argparse.ArgumentParser(description='Implementation of WGR for CT slices dataset')

parser.add_argument('--Xdim', default=520, type=int, help='dimensionality of X')
parser.add_argument('--Ydim', default=5, type=int, help='dimensionality of Y')

parser.add_argument('--noise_dim', default=30, type=int, help='dimensionality of noise vector')
parser.add_argument('--noise_dist', default='gaussian', type=str, help='distribution of noise vector')

parser.add_argument('--train', default=14948, type=int, help='size of train dataset')
parser.add_argument('--val', default=1100, type=int, help='size of validation dataset')
parser.add_argument('--test', default=5000, type=int, help='size of test dataset')

parser.add_argument('--train_batch', default=128, type=int, metavar='BS', help='batch size while training')
parser.add_argument('--val_batch', default=100, type=int, metavar='BS', help='batch size while validation')
parser.add_argument('--test_batch', default=100, type=int, metavar='BS', help='batch size while testing')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train')

args = parser.parse_args()

print(args)

# import data
all_UJI = pd.read_csv("C:/Users/tracy/Desktop/WGR/GitHubRep/WGR/data/UJIndoorLocData.csv")
all_UJI = all_UJI.iloc[:, :-4]

# Extract features and responses before splitting
X_data = all_UJI.iloc[:, :-5].values  # Here, we use 6, because the dimensionality of this dataset is 6.
y_data = all_UJI.iloc[:, -5:].values

# Create and fit scalers
X_scaler = StandardScaler()
X_all_scaled = X_scaler.fit_transform(X_data)

y_scalers = [StandardScaler() for _ in range(5)]
y_scaled = np.zeros_like(y_data)

for i in range(5):
    # Fit and transform each column
    y_scaled[:, i] = y_scalers[i].fit_transform(y_data[:, i].reshape(-1, 1)).flatten()

# Combine scaled responses with features for splitting
all_data_scaled = np.hstack((X_all_scaled, y_scaled))

# Convert back to DataFrame for easier handling
all_UJI_scaled = pd.DataFrame(all_data_scaled)

# Split data into training, validation, and testing datasets
train_val_data, test_data = train_test_split(all_UJI_scaled, test_size=args.test, random_state=5678)
train_data, val_data = train_test_split(train_val_data, test_size=args.val, random_state=5678)

# Convert pandas DataFrames to PyTorch tensors
X_train = torch.tensor(train_data.values[:, :-5], dtype=torch.float32)
y_train = torch.tensor(train_data.values[:, -5:], dtype=torch.float32)

X_val = torch.tensor(val_data.values[:, :-5], dtype=torch.float32)
y_val = torch.tensor(val_data.values[:, -5:], dtype=torch.float32)

X_test = torch.tensor(test_data.values[:, :-5], dtype=torch.float32)
y_test = torch.tensor(test_data.values[:, -5:], dtype=torch.float32)


args.Xdim = get_dimension(X_train)
args.Ydim = get_dimension(y_train)
print(args.Xdim, args.Ydim)

model = Bayesian_fnn(in_dim=520, out_dim=5, hidden_dims=[90, 45])

# Set Pyro random seed
pyro.set_rng_seed(5678)

nuts_kernel = NUTS(model, jit_compile=False)  # jit_compile=True is faster but requires PyTorch 1.6+

    
# Define MCMC sampler, get 50 posterior samples
mcmc = MCMC(nuts_kernel, num_samples=50)
mcmc.run(X_train.float(), y_train.float()) 

predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
preds = predictive(test_X)
    
test_evaluation( test_Y, preds['obs'])

preds['obs'].quantile(0.05,axis=0)
preds['obs'].quantile(0.95,axis=0)

torch.save(predictive,'C:/Users/tracy/Desktop/WGR/BNN/BNN-InoorLoc.pth')

LB  = preds['obs'].quantile(0.05,axis=0)
UB  = preds['obs'].quantile(0.95,axis=0)

CP = BNN_CP(y_test, LB, UB, samples=args.test)

PIL = torch.mean(torch.abs(UB-LB), dim=0)
LB_std = torch.std(y_test-LB, dim=0)
UB_std = torch.std(UB-y_test, dim=0)
