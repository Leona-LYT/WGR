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
from sklearn.preprocessing import StandardScaler

from utils.basic_utils import setup_seed, get_dimension, selection_m
from models.generator import generator_fnn
from models.discriminator import discriminator_fnn
from utils.training_utils import train_WGR_fnn
from utils.evaluation_utils import eva_G_MultiY

import argparse


parser = argparse.ArgumentParser(description='Implementation of WGR for CT slices dataset')

parser.add_argument('--Xdim', default=100, type=int, help='dimensionality of X')
parser.add_argument('--Ydim', default=1, type=int, help='dimensionality of Y')

parser.add_argument('--m_set', default=20, type=int, help='dimensionality of noise vector')
parser.add_argument('--noise_dist', default='gaussian', type=str, help='distribution of noise vector')

parser.add_argument('--train', default=14948, type=int, help='size of train dataset')
parser.add_argument('--val', default=1100, type=int, help='size of validation dataset')
parser.add_argument('--test', default=5000, type=int, help='size of test dataset')

parser.add_argument('--train_batch', default=100, type=int, metavar='BS', help='batch size while training')
parser.add_argument('--val_batch', default=100, type=int, metavar='BS', help='batch size while validation')
parser.add_argument('--test_batch', default=100, type=int, metavar='BS', help='batch size while testing')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train')

args = parser.parse_args()

print(args)

# import data
all_UJI = pd.read_csv("../data/UJIndoorLocData.csv")
all_UJI = all_UJI.iloc[:, :-3]

# Extract features and responses before splitting
X_data = all_UJI.iloc[:, :-6].values
y_data = all_UJI.iloc[:, -6:].values

# Create and fit scalers
X_scaler = StandardScaler()
X_all_scaled = X_scaler.fit_transform(X_data)

y_scalers = [StandardScaler() for _ in range(6)]
y_scaled = np.zeros_like(y_data)


for i in range(6):
    # Fit and transform each column
    y_scaled[:, i] = y_scalers[i].fit_transform(y_data[:, i].reshape(-1, 1)).flatten()


setup_seed(5678)  
# Combine scaled responses with features for splitting
all_data_scaled = np.hstack((X_all_scaled, y_scaled))

# Convert back to DataFrame for easier handling
all_UJI_scaled = pd.DataFrame(all_data_scaled)

# Split data into training, validation, and testing datasets
train_val_data, test_data = train_test_split(all_UJI_scaled, test_size=args.test )
train_data, val_data = train_test_split(train_val_data, test_size=args.val )


# Convert pandas DataFrames to PyTorch tensors
X_train = torch.tensor(train_data.values[:, :-6], dtype=torch.float32)
y_train = torch.tensor(train_data.values[:, -6:], dtype=torch.float32)

X_val = torch.tensor(val_data.values[:, :-6], dtype=torch.float32)
y_val = torch.tensor(val_data.values[:, -6:], dtype=torch.float32)

X_test = torch.tensor(test_data.values[:, :-6], dtype=torch.float32)
y_test = torch.tensor(test_data.values[:, -6:], dtype=torch.float32)


args.Xdim = get_dimension(X_train)
args.Ydim = get_dimension(y_train)


# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
loader_train = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, drop_last=True)
loader_val = DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False, drop_last=True)
loader_test = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, drop_last=True)


# Generate the candidate set for m dynamically
candidate_set = set([1] + [5*i for i in range(1, 21)])
sorted_list = sorted(candidate_set)  # Convert set to a sorted list

G_m_score = []
UJI_results = []


for k in range(args.m_set):
    print('============================ REPLICATION ==============================')
    print("m: " + str(sorted_list[k]))
        
    setup_seed(5678) # set seed to make sure the initialization of networks are the same

    global G_net, D_net, trained_G, trained_D

    # Define generator network and discriminator network
    G_net = generator_fnn(Xdim=args.Xdim, Ydim=args.Ydim, noise_dim=sorted_list[k], hidden_dims = [512, 256, 128, 64])
    D_net = discriminator_fnn(input_dim=args.Xdim+args.Ydim, hidden_dims = [512, 256, 128, 64])

    # Initialize RMSprop optimizers
    D_solver = optim.Adam(D_net.parameters(), lr=0.001, betas=(0.9, 0.999))
    G_solver = optim.Adam(G_net.parameters(), lr=0.001, betas=(0.9, 0.999))   
    
    # Train the model using the train_model function
    trained_G, trained_D = train_WGR_fnn(D=D_net, G=G_net, D_solver=D_solver, G_solver=G_solver, loader_train = loader_train, 
                                     loader_val=loader_val, noise_dim=sorted_list[k], Xdim=args.Xdim, Ydim=args.Ydim, 
                                     lambda_w=0.95, lambda_l=0.05, batch_size=args.train_batch, save_path='./UJI/',  start_eva=100, 
                                     model_type='UJI', device='cpu', num_epochs=50, multivariate=True)
    #evaluate
    m_score = selection_m(G=trained_G, x=X_train, y=y_train, noise_dim=sorted_list[k], Xdim=args.Xdim, Ydim=args.Ydim, train_size=args.train)
    
    print(m_score)
    G_m_score.append(m_score)
     

print("m_score:", G_m_score)

#saving the results as csv
G_m_score_csv = pd.DataFrame(G_m_score)

G_m_score_csv.to_csv("./UJI/G_m_score_UJI_d"+str(args.Xdim)+"_m_select.csv")
 

    






