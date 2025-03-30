def eva_real_G(G, loader_data, Ydim, J_t_size=50):
    """
    Evaluate a generator model on real data.
    
    Parameters:
        G (nn.Module): Generator model
        loader_data (DataLoader): Data loader for evaluation
        Ydim (int): Dimension of output Y
        J_t_size (int): Number of samples to generate for each input
    
    Returns:
        tuple: Mean L1 loss, mean L2 loss, coverage probability, length of prediction interval, 
               standard deviation of upper bound error, standard deviation of lower bound error std
    """
    num_batches = len(loader_data)
    quantiles = [0.025, 0.975]  # Lower and upper bounds for 95% prediction interval
    
    with torch.no_grad():
        val_L1 = torch.zeros([num_batches, Ydim])
        val_L2 = torch.zeros([num_batches, Ydim])
        Q_025 = torch.zeros([num_batches, Ydim])
        Q_975 = torch.zeros([num_batches, Ydim])
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
            val_L1[batch_idx] = torch.mean(torch.abs(output.mean(dim=0) - y), dim=0)
            val_L2[batch_idx] = torch.mean((output.mean(dim=0) - y) ** 2, dim=0)
            
            # Calculate quantiles
            lower_bound = output.quantile(quantiles[0], dim=0)  # [batch_size, Ydim]
            upper_bound = output.quantile(quantiles[1], dim=0)  # [batch_size, Ydim]
            
            # Store quantiles
            Q_025[batch_idx] = lower_bound
            Q_975[batch_idx] = upper_bound
            
            # Calculate length of prediction interval
            LPI[batch_idx] = torch.mean(upper_bound - lower_bound, dim=0)
            
            # Calculate standard deviation of upper and lower bound errors
            SD_UBE[batch_idx] = torch.std(F.mse_loss(upper_bound, y, reduction='none'), dim=0)
            SD_LBE[batch_idx] = torch.std(F.mse_loss(lower_bound, y, reduction='none'), dim=0)
            
            # Calculate coverage probability (whether y is within prediction interval)
            in_interval = (y >= lower_bound) & (y <= upper_bound)
            CP[batch_idx] = in_interval.float().mean(dim=0)
        
        # Calculate average metrics across batches
        mean_val_L1 = val_L1.mean(dim=0)
        mean_val_L2 = val_L2.mean(dim=0)
        mean_CP = CP.mean(dim=0)
        mean_LPI = LPI.mean(dim=0)
        mean_SD_UBE = SD_UBE.mean(dim=0)
        mean_SD_LBE = SD_LBE.mean(dim=0)
        
        # Print results
        print(f"L1 Loss: {mean_val_L1}")
        print(f"L2 Loss: {mean_val_L2}")
        print(f"Coverage Probability: {mean_CP.detach().numpy()}")
        print(f"Length of Prediction Interval: {mean_LPI.detach().numpy()}")
        print(f"SD of Upper Bound Error: {mean_SD_UBE.detach().numpy()}")
        print(f"SD of Lower Bound Error: {mean_SD_LBE.detach().numpy()}")
        
        return (mean_val_L1.detach().numpy(),
                mean_val_L2.detach().numpy(),
                mean_CP.detach().numpy(),
                mean_LPI.detach().numpy(),
                mean_SD_UBE.detach().numpy(),
                mean_SD_LBE.detach().numpy())
