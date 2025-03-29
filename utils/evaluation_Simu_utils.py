def MSE_mean_sd_G_uniY(J_t_size, model_type="M1"):
    with torch.no_grad():
        num_batches = test_size // batch_size
        val_L1 = torch.zeros(num_batches)
        val_L2 = torch.zeros(num_batches)
        mse_mean = torch.zeros(num_batches)
        mse_sd = torch.zeros(num_batches)
        
        for batch_idx, (x, y) in enumerate(loader_test):
            # Generate multiple outputs with different noise
            outputs = torch.stack([
                G(torch.cat([x, sample_noise(x.size(0), args.noise)], dim=1)).view(x.size(0))
                for _ in range(J_t_size)
            ])
            
            # Calculate ground truth mean and standard deviation based on model type
            if model_type == "M1":
                y_test = x[:,0]**2 + torch.exp(x[:,1]+x[:,2]/3) + x[:,3] - x[:,4]
                y_sd = torch.sqrt(0.5 + x[:,1]**2/2 + x[:,4]**2/2)
            elif model_type == "M2":
                beta = torch.tensor([1, 1, -1, -1, 1] + [0]*95).float()
                x_si = x @ beta
                y_test = x_si**2 + torch.sin(x_si.abs()) + 2*torch.exp(-0.5)
                y_sd = torch.sqrt(2*(1+torch.exp(-0.5)))
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
            
            # Calculate output mean and standard deviation
            output_mean = outputs.mean(dim=0)
            output_sd = ((outputs - output_mean)**2).mean(dim=0).sqrt()
            
            # Compute metrics
            val_L1[batch_idx] = l1_loss(output_mean, y)
            val_L2[batch_idx] = l2_loss(output_mean, y)
            mse_mean[batch_idx] = ((output_mean - y_test)**2).mean()
            mse_sd[batch_idx] = ((output_sd - y_sd)**2).mean()
        
        # Calculate average metrics
        metrics = [m.mean().detach().item() for m in [val_L1, val_L2, mse_mean, mse_sd]]
        print(f"L1: {metrics[0]:.4f}, L2: {metrics[1]:.4f}, MSE_mean: {metrics[2]:.4f}, MSE_sd: {metrics[3]:.4f}")
        
        return metrics

# =============================================================================
# MSE_quantile
# =============================================================================
def MSE_quantile_G_uniY(model_type="M1", num_samples=500):
    """
    Evaluate quantile predictions of generator G against true quantiles.
    
    Args:
        model_type: One of "M1", "M2", "SM1", "SM2", "SM3"
        num_samples: Number of samples to draw from G for quantile estimation
    
    Returns:
        Tuple of MSE for 5%, 25%, 50%, 75%, and 95% quantiles
    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    
    with torch.no_grad():
        num_batches = test_size // batch_size
        Q_errors = {q: torch.zeros(num_batches) for q in quantiles}
        
        for batch_idx, (x, y) in enumerate(loader_test):
            # Generate samples from G
            outputs = torch.stack([
                G(torch.cat([x, sample_noise(x.size(0), args.noise)], dim=1)).view(x.size(0))
                for _ in range(num_samples)
            ])
            
            # Calculate true quantiles based on model type
            if model_type == "M1":
                A = x[:,0]**2 + torch.exp(x[:,1]+x[:,2]/3) + x[:,3] - x[:,4]
                B = torch.sqrt(0.5 + x[:,1]**2/2 + x[:,4]**2/2)
                true_quantiles = {q: norm.ppf(q, A, B) for q in quantiles}
                
            elif model_type == "M2":
                beta = torch.tensor([1, 1, -1, -1, 1] + [0]*(args.Xdim-5)).float()
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
                Q_errors[q][batch_idx] = ((predicted_q - true_quantiles[q])**2).mean()
        
        # Calculate average errors
        results = [Q_errors[q].mean().detach().item() for q in quantiles]
        
        # Print results
        result_str = ", ".join([f"Q_{int(q*100)}: {res:.4f}" for q, res in zip(quantiles, results)])
        print(result_str)
        
        return tuple(results)
