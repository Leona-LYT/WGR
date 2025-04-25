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

from utils.basic_utils import setup_seed, get_dimension, selection_m
from models.generator import generator_fnn
from models.discriminator import discriminator_fnn
from utils.training_utils import train_WGR_fnn
from utils.evaluation_utils import eva_G_UniY

import argparse

if 'ipykernel_launcher.py' in sys.argv[0]:  #if not work in jupyter, you can delete this part
    import sys
    sys.argv = [sys.argv[0]] 


parser = argparse.ArgumentParser(description='Implementation of WGR for dataset with one dimensional response Y')

parser.add_argument('--Xdim', default=100, type=int, help='dimensionality of X')
parser.add_argument('--Ydim', default=1, type=int, help='dimensionality of Y')

parser.add_argument('--m_set', default=20, type=int, help='the size of the candidate set of the dimensionality of noise vector')
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
all_CT = pd.read_csv("../data/CT.csv")
all_CT = all_CT.iloc[:, 1:] 


setup_seed(5678)
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

# Generate the candidate set for m dynamically
candidate_set = set( [5*i for i in range(1, 21)])
sorted_list = sorted(candidate_set)  # Convert set to a sorted list

def main():
    G_m_score = []  # Initialize as list
    CT_results = []
   
    for k in range(args.m_set):
        print('============================ REPLICATION ==============================')
        print("m: " + str(sorted_list[k]))
        
        setup_seed(5678) # set seed to make sure the initialization of networks are the same

        global G_net, D_net, trained_G, trained_D

        # Define generator network and discriminator network
        G_net = generator_fnn(Xdim=args.Xdim, Ydim=args.Ydim, noise_dim=sorted_list[k], hidden_dims = [64, 32])
        D_net = discriminator_fnn(input_dim=args.Xdim+args.Ydim, hidden_dims = [64, 32])

        # Initialize RMSprop optimizers
        D_solver = optim.Adam(D_net.parameters(), lr=0.001, betas=(0.9, 0.999))
        G_solver = optim.Adam(G_net.parameters(), lr=0.001, betas=(0.9, 0.999))                    

        # Training
        trained_G, trained_D = train_WGR_fnn(D=D_net, G=G_net, D_solver=D_solver, G_solver=G_solver, loader_train = loader_train, 
                                             loader_val=loader_val, noise_dim=sorted_list[k], Xdim=args.Xdim, Ydim=args.Ydim, J_size=200,
                                             lambda_w=0.2, lambda_l=0.8, batch_size=args.train_batch, save_path='./', 
                                             model_type='CT_m_', device='cpu', num_epochs=100)

         CT_numerical_results = eva_G_UniY(G=G_net, loader_data=loader_test, noise_dim=sorted_list[k], batch_size=args.test_batch, J_t_size=500)
        m_score = selection_m(L2_value = CT_numerical_results[1], noise_dim=sorted_list[k], Xdim=args.Xdim, train_size=args.train)
        #test_G_mean_sd.append(mean_sd_result.detach().cpu().numpy())
        G_m_score.append(m_score)
       
        
    print("m_score:", G_m_score)

    #saving the results as csv
    G_m_score_csv = pd.DataFrame(G_m_score)

    G_m_score_csv.to_csv("./G_m_score_CT_d"+str(args.Xdim)+"_m_selection.csv")
    

    return G_m_score
   
#run
G_m_score= main()

#print selected m
print("selected m:", sorted_list[np.argmin(G_m_score).item()] )
