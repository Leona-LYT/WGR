"""
This is the code for conformal quantile prediction method in the eal data analysis
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

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive

from models.BNN import Bayesian_fnn, BNN_CP
from utils.basic_utils import setup_seed, get_dimension, bnn_evaluation

import argparse

if 'ipykernel_launcher.py' in sys.argv[0]:  #if not work in jupyter, you can delete this part
    import sys
    sys.argv = [sys.argv[0]] 


parser = argparse.ArgumentParser(description='Implementation of WGR for dataset with one dimensional response Y')

parser.add_argument('--Xdim', default=100, type=int, help='dimensionality of X')
parser.add_argument('--Ydim', default=1, type=int, help='dimensionality of Y')

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
all_CT = pd.read_csv("../data/CT.csv")
all_CT = all_CT.iloc[:, 1:] 

setup_seed(5678)  
#split data into training dataset, testing dataset and validation dataset
train_cal_data, test_data = train_test_split(all_CT, test_size=args.test)#, random_state=5678)
train_data, cal_data = train_test_split(train_cal_data, test_size=args.val)#, random_state=5678)

# Convert pandas DataFrames to PyTorch tensors
X_train = torch.tensor(train_data.values[:, :-1], dtype=torch.float32)
y_train = torch.tensor(train_data.values[:, -1], dtype=torch.float32)

X_cal = torch.tensor(cal_data.values[:, :-1], dtype=torch.float32)
y_cal = torch.tensor(cal_data.values[:, -1], dtype=torch.float32)

X_test = torch.tensor(test_data.values[:, :-1], dtype=torch.float32)
y_test = torch.tensor(test_data.values[:, -1], dtype=torch.float32)

args.Xdim = get_dimension(X_train)
args.Ydim = get_dimension(y_train)
print(args.Xdim, args.Ydim)

model = Bayesian_fnn(in_dim=383, out_dim=1, hidden_dims=[40, 20])

# Set Pyro random seed
pyro.set_rng_seed(5678)

nuts_kernel = NUTS(model, jit_compile=False)  # jit_compile=True is faster but requires PyTorch 1.6+

    
# Define MCMC sampler, get 50 posterior samples
mcmc = MCMC(nuts_kernel, num_samples=50)
mcmc.run(X_train.float(), y_train.float()) 
    
predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
preds = predictive(X_test)
    
test_evaluation(X_test, y_test, preds['obs'])

preds['obs'].quantile(0.05,axis=0)
preds['obs'].quantile(0.95,axis=0)

torch.save(predictive,'./BNN-CT.pth')

LB  = preds['obs'].quantile(0.025,axis=0)
UB  = preds['obs'].quantile(0.975,axis=0)

CP = BNN_CP(y_test, LB, UB, samples=args.test)

PIL = torch.mean(torch.abs(UB-LB))
LB_std = torch.std(y_test-LB)
UB_std = torch.std(UB-y_test)





