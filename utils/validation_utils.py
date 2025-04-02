# =============================================================================
# validation
# =============================================================================
def val_G(G, loader_data, noise_dim, Xdim, Ydim, num_samples=100, device='cpu', multivariate=False):
    """
    Validate generator performance using L1 and L2 losses.
    Handles both univariate and multivariate outputs.
    
    Args:
        G: Generator model
        loader_data: Validation data loader
        noise_dim: Dimension of noise vector
        num_samples: Number of samples to generate for each input (default: 100)
        device: Device to run validation on ('cuda' or 'cpu')
        loss_functions: Custom loss functions as dict with 'l1' and 'l2' keys (optional)
        multivariate: Whether the output is multivariate (default: False)
        
    Returns:
        tuple: Mean L1 and L2 losses as numpy values
    """
    G.eval()  # Set generator to evaluation mode
    
    num_batches = len(loader_data)
    
    # Initialize loss tensors based on output dimensionality
    val_L1 = torch.zeros(num_batches, Ydim, device=device)
    val_L2 = torch.zeros(num_batches, Ydim, device=device)
   
    # Reset data loader iterator if we used it to check dimensions
    loader_data_iter = iter(loader_data)
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            try:
                x, y = next(loader_data_iter)
            except StopIteration:
                break
                
            x, y = x.to(device), y.to(device)
            
            # Generate multiple outputs for each input
            outputs = []
            for i in range(num_samples):
                eta = sample_noise(x.size(0), noise_dim).to(device)
                
                # Handle input reshaping if needed
                g_input = torch.cat([x.view(x.size(0), Xdim), eta], dim=1)
                
                # Get generator output
                output = G(g_input)
                
                # Handle reshaping for univariate output if needed
                if not multivariate:
                    output = output.view(x.size(0))
                
                outputs.append(output)
            
            # Stack all outputs and compute mean
            stacked_outputs = torch.stack(outputs, dim=0)
            mean_output = stacked_outputs.mean(dim=0)
            
            # Calculate losses
            if multivariate:
                 val_L1[batch_idx] = torch.mean(torch.abs(mean_output - y), dim=0)
                 val_L2[batch_idx] = torch.mean((mean_output - y)**2, dim=0)
            else:
                 val_L1[batch_idx] = torch.mean(torch.abs(mean_output - y))
                 val_L2[batch_idx] = torch.mean((mean_output - y)**2)
    
    # Compute mean losses across all batches
    if multivariate:
        mean_L1_per_dim = torch.mean(val_L1, dim=0)
        mean_L2_per_dim = torch.mean(val_L2, dim=0)
        print(f"Mean L1 Loss per dimension: {mean_L1_per_dim}")
        print(f"Mean L2 Loss per dimension: {mean_L2_per_dim}")
        scalar_L1 = torch.mean(val_L1).cpu().numpy()
        scalar_L2 = torch.mean(val_L2).cpu().numpy()
    else:
        mean_L1 = val_L1.mean()
        mean_L2 = val_L2.mean()
        print(f"Mean L1 Loss: {mean_L1:.6f}, Mean L2 Loss: {mean_L2:.6f}")
        scalar_L1 = mean_L1.cpu().numpy()
        scalar_L2 = mean_L2.cpu().numpy()
    
    return scalar_L1, scalar_L2
