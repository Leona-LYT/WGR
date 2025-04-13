import torch
from scipy.stats import norm
from scipy.stats import lognorm
import math
import torch.nn.functional as F
from utils.basic_utils import sample_noise
from data.SimulationData import DataGenerator, generate_multi_responses_multiY

# =============================================================================
# L1 and L2 error, MSE of conditional mean and conditional standard deviation
# =============================================================================
def L1L2_MSE_mean_sd_G(G, loader_dataset, noise_dim, Xdim, test_size, batch_size, J_t_size=50, model_type="M1", Ydim=1, is_multivariate=False):
    """
    Calculate L1 and L2 error ,MSE of the conditional mean and conditional standard deviation for both univariate and multivariate Y.
    
    Parameters:
        G (torch.nn.Module): Generator model
        loader_dataset (DataLoader): Data loader for test data
        noise_dim: Dimension of noise vector
        Xdim: Dimension of X
        test size: The sample size of test dataset
        batch_size: Batch size
        J_t_size (int): Number of samples to generate for each input
        model_type (str): Type of model to evaluate ('M1', 'M2', 'M3', 'M4', 'SM1', 'SM2', 'SM3', 'SM4')
        Ydim (int): Dimension of the output Y (1 for univariate, >1 for multivariate)
        is_multivariate (bool): Whether to use multivariate evaluation logic
    
    Returns:
        tuple: Mean L1 loss, mean L2 loss, MSE of mean, MSE of standard deviation
    """
    with torch.no_grad():
        
        num_batches = test_size // batch_size
        
        # Initialize metrics tensors with appropriate dimensions
        eva_L1 = torch.zeros(num_batches, Ydim)
        eva_L2 = torch.zeros(num_batches, Ydim)
        mse_mean = torch.zeros(num_batches, Ydim)
        mse_sd = torch.zeros(num_batches, Ydim)
        
        for batch_idx, (x, y) in enumerate(loader_dataset):
            # Generate samples - method depends on whether it's multivariate or not
            if is_multivariate:
                output = torch.zeros([J_t_size, batch_size, Ydim])
                for i in range(J_t_size):
                    eta = sample_noise(x.size(0), noise_dim)
                    g_input = torch.cat([x.view(x.size(0), 1), eta], dim=1)
                    output[i] = G(g_input).detach()
            else:
                # More efficient method for univariate case using list comprehension
                outputs = torch.stack([
                    G(torch.cat([x, sample_noise(x.size(0), noise_dim)], dim=1)).view(x.size(0))
                    for _ in range(J_t_size)
                ])
                
            # Calculate sample statistics
            if is_multivariate:
                output_mean = torch.mean(output,dim=0)
                output_sd = ((output - output_mean)**2).mean(dim=0).sqrt()
            else:
                output_mean = outputs.mean(dim=0)
                output_sd = ((outputs - output_mean)**2).mean(dim=0).sqrt()
            
            # Set conditional mean and standard deviation based on model type
            if model_type == "M1":
                y_test = x[:,0]**2 + torch.exp(x[:,1]+x[:,2]/3) + x[:,3] - x[:,4]
                y_sd = torch.sqrt(0.5 + x[:,1]**2/2 + x[:,4]**2/2)
            elif model_type == "M2":
                beta = torch.tensor([1, 1, -1, -1, 1] + [0]*(Xdim-5)).float()
                x_si = x @ beta
                y_test = x_si**2 + torch.sin(x_si.abs()) + 2*torch.exp(torch.tensor(-0.5))
                y_sd = torch.sqrt( 2-4*torch.exp(torch.tensor(-1)) +2* torch.exp(torch.tensor(-2))  )
            elif model_type == "SM1":
                y_test = x[:,0]**2 + torch.exp(x[:,1]+x[:,2]/3) + torch.sin(x[:,3] + x[:,4])
                y_sd = torch.ones_like(y_test)
            elif model_type == "SM2":
                y_test = x[:,0]**2 + torch.exp(x[:,1]+x[:,2]/3) + x[:,3] - x[:,4]
                y_sd = torch.full_like(y_test, np.sqrt(3))
            elif model_type == "SM3":
                A = 5 + x[:,0]**2/3 + x[:,1]**2 + x[:,2]**2 + x[:,3] + x[:,4]
                y_test = A * np.exp(0.0625)
                y_sd = torch.sqrt(A**2 * (np.exp(0.25) - np.exp(0.125)))

            if is_multivariate:
                Y_generate = torch.zeros([500,x.size(0),Ydim])
                for j in range(x.size(0)):
                    Y_generate[:,j,:] = generate_multi_responses_multiY(x_value = x[j].view([1]).item(), n_responses=500,model_type=model_type)
                    y_test = torch.mean(Y_generate,dim=0) 
                    y_sd = torch.std(Y_generate,dim=0)  

            
            # Calculate metrics based on whether it's multivariate or not
            if is_multivariate:
                eva_L1[batch_idx] = torch.mean(torch.abs(output_mean - y), dim=0)
                eva_L2[batch_idx] = torch.mean((output_mean - y)**2, dim=0)
                mse_mean[batch_idx] = torch.mean((output_mean - y_test)**2, dim=0)
                mse_sd[batch_idx] = torch.mean((output_sd - y_sd)**2, dim=0)
            else:
                eva_L1[batch_idx] = torch.mean(torch.abs(output_mean - y))
                eva_L2[batch_idx] = torch.mean((output_mean - y)**2)
                mse_mean[batch_idx] = torch.mean((output_mean - y_test)**2)
                mse_sd[batch_idx] = torch.mean((output_sd - y_sd)**2)
        
        # Calculate mean metrics across batches
        mean_eva_L1 = torch.mean(eva_L1, dim=0)
        mean_eva_L2 = torch.mean(eva_L2, dim=0)
        mean_mse_mean = torch.mean(mse_mean, dim=0)
        mean_mse_sd = torch.mean(mse_sd, dim=0)
        
        # Print results
        print(f"Model: {model_type}, {'Multivariate' if is_multivariate else 'Univariate'}, Ydim: {Ydim}, J_t_size: {J_t_size}")
        print(f"L1 Loss: {mean_eva_L1}")
        print(f"L2 Loss: {mean_eva_L2}")
        print(f"MSE Mean: {mean_mse_mean}")
        print(f"MSE SD: {mean_mse_sd}")
        
        # Return means across all dimensions
        return mean_eva_L1.detach().numpy(), mean_eva_L2.detach().numpy(), mean_mse_mean.detach().numpy(), mean_mse_sd.detach().numpy()
        
# =============================================================================
# MSE of conditional quantile
# =============================================================================
def MSE_quantile_G_uniY(G, loader_dataset,  noise_dim, Xdim, test_size, batch_size, J_t_size=500, model_type="M1" ):
    """
    Evaluate the MSE of the conditional quantiles generated by G against true quantiles at different levels
    
    Args:
        G (nn.Module): Generator model
        loader_dataset (DataLoader): Data loader for test data
        noise_dim: Dimension of noise vector
        test size: The sample size of test dataset
        batch_size: Batch size
        model_type: One of "M1", "M2", "SM1", "SM2", "SM3"
        J_t_size: Number of samples to draw from G for quantile estimation
    
    Returns:
        Tuple of MSE for 5%, 25%, 50%, 75%, and 95% quantiles
    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95] #default quantile levels
    
    with torch.no_grad():
        num_batches = test_size // batch_size
        Q_errors = {q: torch.zeros(num_batches) for q in quantiles}
        
        for batch_idx, (x, y) in enumerate(loader_dataset):
            # Generate samples from G
            outputs = torch.stack([
                G(torch.cat([x, sample_noise(x.size(0), noise_dim)], dim=1)).view(x.size(0))
                for _ in range(J_t_size)
            ])
            
            # Calculate true quantiles based on model type
            if model_type == "M1":
                A = x[:,0]**2 + torch.exp(x[:,1]+x[:,2]/3) + x[:,3] - x[:,4]
                B = torch.sqrt(0.5 + x[:,1]**2/2 + x[:,4]**2/2)
                true_quantiles = {q: norm.ppf(q, A, B) for q in quantiles}
                
            elif model_type == "M2":
                beta = torch.tensor([1, 1, -1, -1, 1] + [0]*(Xdim-5)).float()
                SI = x @ beta
                
                # Simulate true distribution for M2
                F_output = torch.stack([
                    SI**2 + torch.sin(torch.abs(SI)) + 2*torch.cos(torch.randn(x.size(0)))
                    for _ in range(10000)
                ])
                
                true_quantiles = {q: F_output.quantile(q, dim=0) for q in quantiles}
                
            elif model_type == "SM1":
                A = x[:,0]**2 + torch.exp(x[:,1]+x[:,2]/3) + torch.sin(x[:,3] + x[:,4])
                true_quantiles = {q: norm.ppf(q, A, 1) for q in quantiles}
                
            elif model_type == "SM2":
                A = x[:,0]**2 + torch.exp(x[:,1]+x[:,2]/3) + x[:,3] - x[:,4]
                true_quantiles = {q: t.ppf(q, 3) + A for q in quantiles}
                
            elif model_type == "SM3":
                A = 5 + x[:,0]**2/3 + x[:,1]**2 + x[:,2]**2 + x[:,3] + x[:,4]
                true_quantiles = {q: lognorm.ppf(q, np.sqrt(0.125), scale=A) for q in quantiles}
            
            # Calculate predicted quantiles and errors
            for q in quantiles:
                predicted_q = outputs.quantile(q, dim=0)
                Q_errors[q][batch_idx] = torch.mean(torch.pow(predicted_q -true_quantiles[q],2))
        
        # Calculate average errors
        results = [Q_errors[q].mean().detach().item() for q in quantiles]
        
        # Print results
        result_str = ", ".join([f"Q_{int(q*100)}: {res:.4f}" for q, res in zip(quantiles, results)])
        print(result_str)
        
        return tuple(results)



def MSE_quantile_G_multiY(G, loader_dataset,  Ydim, Xdim, noise_dim, batch_size, test_size, model_type, J_t_size=500):
    """
    Calculate quantile metrics for multivariate generator outputs.
    
    Parameters:
        G (nn.Module): Generator model
        loader_dataset (DataLoader): Data loader for test data
        Ydim (int): Dimension of the output Y (default: 2)
        noise_dim: Dimension of noise vector
        test_size: The sample size of test dataset
        batch_size (int): Batch size
        J_t_size (int): Number of samples to generate for each input
        model_type (str): Type of model for generating responses
        
    
    Returns:
        torch.Tensor: Stacked quantile metrics of shape [num_quantiles, Ydim]
    """
    quantiles=[0.05, 0.25, 0.50, 0.75, 0.95] #default quantile levels
    
    with torch.no_grad():
        num_batches = test_size // batch_size
        
        # Initialize metrics for each quantile
        Q_metrics = {q: torch.zeros(num_batches, Ydim) for q in quantiles}
        
        for batch_idx, (x, y) in enumerate(loader_dataset):
            # Generate samples from the model
            output = torch.zeros([J_t_size, batch_size, Ydim])
            for i in range(J_t_size):
                eta = sample_noise(x.size(0), noise_dim)
                g_input = torch.cat([x.view(x.size(0), 1), eta], dim=1)
                output[i] = G(g_input).detach()
            
            # Generate ground truth responses
            generate_Y = torch.zeros([J_t_size, batch_size, Ydim])
            for j in range(batch_size):
                # Generate multiple responses for each x value
                generate_Y[:, j, :] = generate_multi_responses_multiY(
                    x[j].view([1]).item(), 
                    n_responses=J_t_size, 
                    model_type=model_type
                )
            
            # Calculate metrics for each quantile
            for q in quantiles:
                # Calculate the model quantiles and ground truth quantiles
                model_quantile = output.quantile(q, dim=0)  # shape: [batch_size, Ydim]
                true_quantile = generate_Y.quantile(q, dim=0)  # shape: [batch_size, Ydim]
                
                # Calculate the mean squared error for each dimension
                Q_metrics[q][batch_idx] = torch.mean((model_quantile - true_quantile)**2, dim=0)
        
        # Calculate average metrics across batches
        Q_result = [torch.mean(Q_metrics[q], dim=0) for q in quantiles]
        
        # Print the results
        print(f"Quantile evaluation for model: {model_type}, Ydim: {Ydim}")
        for i, q in enumerate(quantiles):
            print(f"Quantile {q:.2f}: {Q_result[i]}")
        
        # Stack and return results
        Q_stacked = torch.stack(Q_result).detach()
        return Q_stacked

# =============================================================================
# evaluation on real data analysis
# =============================================================================
def eva_real_G(G, loader_data, Ydim, noise_dim, batch_size, J_t_size=50):
    """
    Evaluate a generator model on real data analysis.
    
    Parameters:
        G (nn.Module): Generator model
        loader_data (DataLoader): Data loader for evaluation
        Ydim (int): Dimension of output Y
        noise_dim (int): Dimension of noise vector eta
        batch_size (int): the batch size of dataset loaded
        J_t_size (int): Number of samples to generate for each input
    
    Returns:
        tuple: Mean L1 loss, mean L2 loss, coverage probability, length of prediction interval, 
               standard deviation of upper bound error, standard deviation of lower bound error std
    """
    num_batches = len(loader_data)
    quantiles = [0.025, 0.975]  # Lower and upper bounds for 95% prediction interval
    
    with torch.no_grad():
        eva_L1 = torch.zeros([num_batches, Ydim])
        eva_L2 = torch.zeros([num_batches, Ydim])
        Q_025 = torch.zeros([num_batches, batch_size])
        Q_975 = torch.zeros([num_batches, batch_size])
        CP = torch.zeros([num_batches, Ydim])
        LPI = torch.zeros([num_batches, Ydim])
        SD_UBE = torch.zeros([num_batches, Ydim])
        SD_LBE = torch.zeros([num_batches, Ydim])
        
        for batch_idx, (x, y) in enumerate(loader_data):
            # Generate multiple samples
            outputs = []
            for i in range(J_t_size):
                eta = sample_noise(x.size(0), noise_dim)
                g_input = torch.cat([x, eta], dim=1).float()
                outputs.append(G(g_input).view(x.size(0), Ydim).detach())
            
            # Stack outputs into a tensor of shape [J_t_size, batch_size, Ydim]
            output = torch.stack(outputs)
            
            # Calculate metrics
            eva_L1[batch_idx] = torch.mean(torch.abs(output.mean(dim=0) - y))
            eva_L2[batch_idx] = torch.mean((output.mean(dim=0) - y) ** 2)
            
            # Calculate quantiles
            lower_bound = output.quantile(quantiles[0], dim=0)  # [batch_size, Ydim]
            upper_bound = output.quantile(quantiles[1], dim=0)  # [batch_size, Ydim]
            
            # Store quantiles
            Q_025[batch_idx] = lower_bound.squeeze(1)
            Q_975[batch_idx] = upper_bound.squeeze(1)
            
            # Calculate length of prediction interval
            LPI[batch_idx] = torch.mean(upper_bound - lower_bound, dim=0)
            
            # Calculate standard deviation of upper and lower bound errors
            SD_UBE[batch_idx] = torch.std(F.mse_loss(upper_bound.squeeze(1), y, reduction='none'), dim=0)
            SD_LBE[batch_idx] = torch.std(F.mse_loss(lower_bound.squeeze(1), y, reduction='none'), dim=0)
            
            # Calculate coverage probability (whether y is within prediction interval)
            in_interval = (y >= lower_bound.squeeze(1)) & (y <= upper_bound.squeeze(1))
            CP[batch_idx] = in_interval.float().mean(dim=0).view([1])
        
        # Calculate average metrics across batches
        mean_eva_L1 = eva_L1.mean(dim=0)
        mean_eva_L2 = eva_L2.mean(dim=0)
        mean_CP = CP.mean(dim=0)
        mean_LPI = LPI.mean(dim=0)
        mean_SD_UBE = SD_UBE.mean(dim=0)
        mean_SD_LBE = SD_LBE.mean(dim=0)
        
        # Print results
        print(f"L1 Loss: {mean_eva_L1}")
        print(f"L2 Loss: {mean_eva_L2}")
        print(f"Coverage Probability: {mean_CP.detach().numpy()}")
        print(f"Length of Prediction Interval: {mean_LPI.detach().numpy()}")
        print(f"SD of Upper Bound Error: {mean_SD_UBE.detach().numpy()}")
        print(f"SD of Lower Bound Error: {mean_SD_LBE.detach().numpy()}")
        
        return (mean_eva_L1.detach().numpy(),
                mean_eva_L2.detach().numpy(),
                mean_CP.detach().numpy(),
                mean_LPI.detach().numpy(),
                mean_SD_UBE.detach().numpy(),
                mean_SD_LBE.detach().numpy())
