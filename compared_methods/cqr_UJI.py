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
from sklearn.preprocessing import StandardScaler

from utils.cqr_utils import quantile_loss, train_quantile_net, conformal_calibration, predict_intervals
from models.regression_net import regression_net
from utils.basic_utils import setup_seed, get_dimension

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
parser.add_argument('--cal_batch', default=100, type=int, metavar='BS', help='batch size while calibration')
parser.add_argument('--test_batch', default=100, type=int, metavar='BS', help='batch size while testing')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train')

args = parser.parse_args()

print(args)

# import data
all_UJI = pd.read_csv("../data/UJIndoorLocData.csv")
all_UJI = all_UJI.iloc[:, :-4]

# Extract features and responses before splitting
X_data = all_UJI.iloc[:, :-5].values
y_data = all_UJI.iloc[:, -5:].values

# Create and fit scalers
X_scaler = StandardScaler()
X_all_scaled = X_scaler.fit_transform(X_data)

y_scalers = [StandardScaler() for _ in range(5)]
y_scaled = np.zeros_like(y_data)

for i in range(5):
    # Fit and transform each column
    y_scaled[:, i] = y_scalers[i].fit_transform(y_data[:, i].reshape(-1, 1)).flatten()

setup_seed(5678)  
# Combine scaled responses with features for splitting
all_data_scaled = np.hstack((X_all_scaled, y_scaled))

# Convert back to DataFrame for easier handling
all_UJI_scaled = pd.DataFrame(all_data_scaled)

# Split data into training, validation, and testing datasets
train_cal_data, test_data = train_test_split(all_UJI_scaled, test_size=args.test )
train_data, cal_data = train_test_split(train_cal_data, test_size=args.val )

# Convert pandas DataFrames to PyTorch tensors
X_train = torch.tensor(train_data.values[:, :-5], dtype=torch.float32)
y_train = torch.tensor(train_data.values[:, -5:], dtype=torch.float32)

X_cal = torch.tensor(cal_data.values[:, :-5], dtype=torch.float32)
y_cal = torch.tensor(cal_data.values[:, -5:], dtype=torch.float32)

X_test = torch.tensor(test_data.values[:, :-5], dtype=torch.float32)
y_test = torch.tensor(test_data.values[:, -5:], dtype=torch.float32)

args.Xdim = get_dimension(X_train)
args.Ydim = get_dimension(y_train)
print(args.Xdim, args.Ydim)

#conduct conformal prediction
cqr_net =  regression_net(in_dim=args.Xdim, out_dim=10, hidden_dims = [128, 64])
optimizer = optim.Adam(cqr_net.parameters(), lr=0.001, betas=(0.9, 0.999))
trained_net = train_quantile_net(model=cqr_net, optimizer=optimizer, X_train=X_train, y_train=y_train, alpha=0.05,epochs=1500,Ydim=args.Ydim)
#calibration
corrections = conformal_calibration(trained_net, X_cal, y_cal, alpha=0.05,Ydim=args.Ydim)
lower_bounds, upper_bounds = predict_intervals(trained_net, X_test, corrections, Ydim=args.Ydim)

#compute LPI and CP
CP,LPI = compute_CP(y_test, lower_bounds, upper_bounds, Ydim=args.Ydim)

std_UBE = torch.std(torch.Tensor(upper_bounds)-y_test,dim=0)
std_LBE = torch.std(torch.Tensor(lower_bounds)-y_test,dim=0)
