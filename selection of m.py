import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from itertools import cycle
import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from pandas import DataFrame

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
parser = argparse.ArgumentParser(description='PyTorch Implementation m selection')

parser.add_argument('--train', default=5000, type=int, help='size of train dataset')
parser.add_argument('--val', default=1000, type=int, help='size of validation dataset')
parser.add_argument('--test', default=1000, type=int, help='size of test dataset')


parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train')
parser.add_argument('--batch', default=128, type=int, metavar='BS', help='batch size')

parser.add_argument('--Xdim', default=5, type=int, help='dimensionality of X')
parser.add_argument('--Ydim', default=1, type=int, help='dimensionality of Y')
parser.add_argument('--reps', default=1, type=int, help='number of replications') #in the selection of m, we only consider one replication

args, unknown = parser.parse_known_args()

beta = torch.tensor([1, 1, -1, -1, 1]).float()
#beta = torch.tensor([1, 1, -1, -1, 1] + [0]*95).float()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device) 

#build a dataset, which is used to train the models
class TensorDataset(Dataset):
    
    "Define the class which is used to build the new data set"
    
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# Generate the candidate set for m dynamically
candidate_set = set(range(args.Ydim, args.Ydim + 20))  # {q, q+1, ..., q+19}; the size of candidate set is 20 
sorted_list = sorted(candidate_set)  # Convert set to a sorted list

# =============================================================================
# noise generation
# =============================================================================
def sample_noise(size, dim):
    
    temp = torch.randn(size, dim)

    return temp

# =============================================================================
# define networks
# =============================================================================
def discriminator():
    model = nn.Sequential(
       # Flatten(),
        nn.Linear(args.Xdim + 1, 64),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(64, 32),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(32, 1) #the probability of true or false
    )
    return model

#define the generator  
def generator(noise):
    model = nn.Sequential(
        nn.Linear(args.Xdim + noise, 64),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(64, 32),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(32, args.Ydim)# the dimension of Y is 1
    )
    return model

# =============================================================================
# define the loss function
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

def reset_grad():
    D_solver.zero_grad()
    G_solver.zero_grad()

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
# validation in terms of L2_prediction
# =============================================================================
def val_G(noise):
    with torch.no_grad():    
        val_L2 = torch.zeros(10)
        
        for batch_idx, (x, y) in enumerate(loader_val):
            output = torch.zeros([200, 100])
            for i in range(200):
                eta = sample_noise(x.size(0), noise)
                g_input = torch.cat([x, eta], dim=1)
                output[i] = G(g_input).view(x.size(0)).detach()
        
            val_L2[batch_idx] = l2_loss(output.mean(dim=0), y)
            
        print(val_L2.mean())   
        return val_L2.mean().detach().numpy() 

# =============================================================================
# test the performance on different m
# =============================================================================
def test_G(noise):
    with torch.no_grad():
        output = torch.zeros([200, args.train])
        F_output = torch.zeros([10000,args.train])
        for i in range(200):
            eta = sample_noise(args.train, noise)
            g_input = torch.cat([train_X, eta], dim=1)
            output[i] = G(g_input).view(-1).detach()
        
        for i in range(10000):
            #compute true F
            eps = torch.randn([args.train])
            F_output[i] = train_SI**2 + torch.sin(torch.abs(train_SI)) + 2*torch.cos(eps) #- eps
        m_value = torch.mean((torch.mean(output,dim=0) - torch.mean(F_output,dim=0) )**2, dim=0)
        
        print(m_value)
        return m_value.detach().item()



def Quantile_G(noise):
    with torch.no_grad():
        Q_05 = torch.zeros(10)
        Q_25 = torch.zeros(10)
        Q_50 = torch.zeros(10)
        Q_75 = torch.zeros(10)
        Q_95 = torch.zeros(10)

        for batch_idx, (x, y) in enumerate(loader_test):
            output = torch.zeros([200, 100])
            F_output = torch.zeros([10000,100])
            for i in range(200):
                eta = sample_noise(x.size(0), noise)
                g_input = torch.cat([x, eta], dim=1)
                output[i] = G(g_input).view(x.size(0)).detach()

            SI = x @ beta
            for i in range(10000):
                #compute true F
                eps = torch.randn([100])
                F_output[i] = SI**2 + torch.sin(torch.abs(SI)) + 2*torch.cos(eps) #- eps

            A = x[:, 0]**2 + torch.exp(x[:, 1] + x[:, 2]/3) + x[:, 3] - x[:, 4]
            output = np.asarray(output)
            F_output = np.asarray(F_output)
            Q_05[batch_idx] = torch.tensor(((np.quantile(output, 0.05, axis=0) - np.quantile(F_output, 0.05,axis=0) )**2).mean())
            Q_25[batch_idx] = torch.tensor(((np.quantile(output, 0.25, axis=0) - np.quantile(F_output, 0.25,axis=0) )**2).mean())
            Q_50[batch_idx] = torch.tensor(((np.quantile(output, 0.50, axis=0) - np.quantile(F_output, 0.50,axis=0) )**2).mean())
            Q_75[batch_idx] = torch.tensor(((np.quantile(output, 0.75, axis=0) - np.quantile(F_output, 0.75,axis=0) )**2).mean())
            Q_95[batch_idx] = torch.tensor(((np.quantile(output, 0.95, axis=0) - np.quantile(F_output, 0.95,axis=0) )**2).mean())
            
        print(Q_05.mean(), Q_25.mean(), Q_50.mean(), Q_75.mean(), Q_95.mean())
        return Q_05.mean().detach().item(), Q_25.mean().detach().item(), Q_50.mean().detach().item(), Q_75.mean().detach().item(), Q_95.mean().detach().item()

# =============================================================================
# train the model
# =============================================================================
def train_gan(D, G, D_solver, G_solver, noise=args.Ydim, num_epochs=10):
    iter_count = 0 
    l2_Acc = val_G(noise)
    Best_acc = l2_Acc.copy()
    
    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(loader_label):
            if x.size(0) != args.batch:
                continue
    
            eta = Variable(sample_noise(x.size(0), noise))
            
            d_input = torch.cat([x, y.view(args.batch, 1)], dim=1)
            g_input = torch.cat([x, eta], dim=1)
            
            #train D
            D_solver.zero_grad()
            logits_real = D(d_input)
            
            fake_y = G(g_input).detach()
            fake_images = torch.cat([x, fake_y.view(args.batch, 1)], dim=1)
            logits_fake = D(fake_images)
            
            penalty = calculate_gradient_penalty(D, logits_real, fake_images, device)
            d_error = discriminator_loss(logits_real, logits_fake) + 10 * penalty
            d_error.backward() 
            D_solver.step()
            ######################################################            
            # train G
            G_solver.zero_grad()
            fake_y = G(g_input)
            fake_images = torch.cat([x, fake_y.view(args.batch, 1)], dim=1)
            logits_fake = D(fake_images)
            
            g_output = torch.zeros([50, args.batch])
            for i in range(50):
                eta = sample_noise(x.size(0), noise)
                g_input = torch.cat([x, eta], dim=1)
                g_output[i] = G(g_input).view(x.size(0))
            
            g_error = 0.9 * generator_loss(logits_fake) + 0.1 * l2_loss(g_output.mean(dim=0), y)
            g_error.backward()
            G_solver.step()
            
            if(iter_count > 2500):    
                if(iter_count % 100 == 0):
                    print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_error.item(), g_error.item()))
                    l2_Acc = val_G(noise)
                    if l2_Acc < Best_acc:
                        print('################## save G model #################')
                        Best_acc = l2_Acc.copy()
                        torch.save(G, './M2d5/WGR-G-m' + str(noise) + '.pth')
                        #torch.save(D, './M2d5/WGR-D-m' + str(noise) + '.pth')
            iter_count += 1

# =============================================================================
# start selections
# =============================================================================
#test_G_reps = torch.zeros([20])  # the size of candidate set
#test_G_quantile = torch.zeros([20, 5])

setup_seed(1234)
reps = (20,)
seed = torch.randint(0, 10000, reps)

setup_seed(seed[0].detach().numpy().item())
        
# =============================================================================
# generate data: M2 in main text
# =============================================================================

# training data
train_X = torch.randn([args.train,args.Xdim])
train_eps = torch.randn([args.train])
train_SI = train_X @ beta
train_Y = train_SI**2 + torch.sin(torch.abs(train_SI)) + 2*torch.cos(train_eps) #- train_eps

# validation data 
val_X = torch.randn([args.val,args.Xdim])
val_eps = torch.randn([args.val])
val_SI = val_X @ beta
val_Y = val_SI**2 + torch.sin(torch.abs(val_SI)) + 2*torch.cos(val_eps) #- val_eps

# test dat
test_X = torch.randn([args.test,args.Xdim])
test_eps = torch.randn([args.test])
test_SI = test_X @ beta
test_Y = test_SI**2 + torch.sin(torch.abs(test_SI)) + 2*torch.cos(test_eps) #- test_eps

train_dataset = TensorDataset(train_X.float(), train_Y.float())
loader_label = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        
val_dataset = TensorDataset(val_X.float(), val_Y.float())
loader_val = DataLoader(val_dataset, batch_size=100, shuffle=True)

test_dataset = TensorDataset(test_X.float(), test_Y.float())
loader_test = DataLoader(test_dataset, batch_size=100, shuffle=True)


def main():
    test_G_reps = []  # Initialize as list
    test_G_quantile = []  # Initialize as list

    for k in range(20):
        
        print('============================ REPLICATION ==============================')
        print(k, seed[0])
        print("m: " + str(sorted_list[k]))

        setup_seed(seed[0].detach().numpy().item())
        
        global D, G, D_solver, G_solver
        
        D = discriminator()
        G = generator(noise=sorted_list[k])

        D_solver = optim.RMSprop(D.parameters(), lr=0.001)
        G_solver = optim.RMSprop(G.parameters(), lr=0.001)

        train_gan(D, G, D_solver, G_solver, noise=sorted_list[k], num_epochs=300)

        G = torch.load('./M2d5/WGR-G-m' + str(sorted_list[k]) + '.pth', weights_only=False)
        
        # Store results
        test_G_reps.append(test_G(noise=sorted_list[k]))  # Append test G reps
        
        quantile_result = Quantile_G(noise=sorted_list[k])
        
        if isinstance(quantile_result, torch.Tensor):
            test_G_quantile.append(quantile_result.detach().cpu().numpy())  # Convert tensor to NumPy array
        else:
            test_G_quantile.append(np.array(quantile_result))  # Convert to NumPy if it's not a tensor
        # Print intermediate results for debugging
        print(f"Test G results at k={k}: {test_G_reps[-1]}")
        print(f"Test G quantile at k={k}: {test_G_quantile[-1]}")
        
    print("Test G results:", test_G_reps)
    print("Test G quantile:", test_G_quantile)

    #save csv
    test_G_reps_csv = pd.DataFrame(test_G_reps)
    test_G_quantile_csv = pd.DataFrame(test_G_quantile)

    test_G_reps_csv.to_csv("./M2d5/test_G_reps.csv")
    test_G_quantile_csv.to_csv("./M2d5/test_G_quantile.csv")

#run
main()

