# =============================================================================
#validation
# =============================================================================
def val_G(G, loader_val, noise_dim, num_samples=100, device='cuda', 
          loss_functions=None, multivariate=False, reshape_input=False):
    """
    Validate generator performance using L1 and L2 losses.
    Handles both univariate and multivariate outputs.
    
    Args:
        G: Generator model
        loader_val: Validation data loader
        noise_dim: Dimension of noise vector
        num_samples: Number of samples to generate for each input (default: 100)
        device: Device to run validation on ('cuda' or 'cpu')
        loss_functions: Custom loss functions as dict with 'l1' and 'l2' keys (optional)
        multivariate: Whether the output is multivariate (default: False)
        reshape_input: Whether to reshape input with .view(x.size(0), 1) (default: False)
        
    Returns:
        tuple: Mean L1 and L2 losses as numpy values
    """
    G.eval()  # Set generator to evaluation mode
    
    num_batches = len(loader_val)
    
    # Initialize loss tensors based on output dimensionality
    if multivariate:
        # For the first batch, determine output dimensionality
        sample_x, sample_y = next(iter(loader_val))
        output_dim = sample_y.size(1)
        val_L1 = torch.zeros(num_batches, output_dim, device=device)
        val_L2 = torch.zeros(num_batches, output_dim, device=device)
    else:
        val_L1 = torch.zeros(num_batches, device=device)
        val_L2 = torch.zeros(num_batches, device=device)
    
    # Reset data loader iterator if we used it to check dimensions
    loader_val_iter = iter(loader_val)
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            try:
                x, y = next(loader_val_iter)
            except StopIteration:
                break
                
            x, y = x.to(device), y.to(device)
            
            # Generate multiple outputs for each input
            outputs = []
            for i in range(num_samples):
                eta = sample_noise(x.size(0), noise_dim).to(device)
                
                # Handle input reshaping if needed
                if reshape_input:
                    g_input = torch.cat([x.view(x.size(0), 1), eta], dim=1)
                else:
                    g_input = torch.cat([x, eta], dim=1)
                
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
            if loss_functions is not None:
                # Use provided loss functions
                val_L1[batch_idx] = loss_functions['l1'](mean_output, y)
                val_L2[batch_idx] = loss_functions['l2'](mean_output, y)
            else:
                # Use default loss calculations
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
