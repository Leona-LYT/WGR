def train_WGR_fnn(D, G, D_solver, G_solver, loader_train, loader_val, noise_dim, Ydim, 
                  batch_size, rep_count, J_size=50, noise_distribution='gaussian', multivariate=False,
                  lambda_w=0.9, lambda_l=0.1, save_path='./M1/', model_type="M1", start_eva=1000,  eva_iter = 50,
                  num_epochs=10, num_samples=100, device='cuda', lr_decay=None, 
                  lr_decay_step=None, lr_decay_gamma=0.1):
    """
    Train Wasserstein GAN Regression with Fully-Connected Neural Networks.
    
    Args:
        D: Discriminator model
        G: Generator model
        D_solver: Discriminator optimizer
        G_solver: Generator optimizer
        loader_train: Data loader for training set
        loader_val: Data loader for validation set
        noise_dim: Dimension of noise vector
        Ydim: Dimension of output Y
        batch_size: Batch size
        rep_count: Repetition count
        J_size: Generator projection size (default: 50)
        noise_distribution: Distribution for noise sampling (default: 'gaussian')
        lambda_w: Weight for Wasserstein loss (default: 0.9)
        lambda_l: Weight for L2 regularization (default: 0.1)
        save_path: Path to save models (default: './M1/')
        start_eva: Iteration to start evaluation (default: 1000)
        eva_iter: to conduct the validation per iteration (default: 50)
        num_epochs: Number of training epochs (default: 10)
        num_samples: Number of noise samples generated for each data point in validation (default: 100)
        device: Device to train on (default: 'cuda')
        lr_decay: Learning rate decay strategy ('step', 'plateau', 'cosine', or None)
        lr_decay_step: Step size for StepLR or patience for ReduceLROnPlateau
        lr_decay_gamma: Multiplicative factor for learning rate decay
    
    Returns:
        tuple: Best validation scores and final models
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Move models to device
    D = D.to(device)
    G = G.to(device)
    
    # Initialize counters and metrics
    iter_count = 0
    l1_acc, l2_acc = val_G(G=G, loader_data=loader_val, noise_dim=noise_dim, Ydim=Ydim, num_samples=num_samples, device=device,  multivariate=multivariate, reshape_input=False)
    
    # Save initial model state
    best_acc = l2_acc
    best_model_g = copy.deepcopy(G.state_dict())
    best_model_d = copy.deepcopy(D.state_dict())
    
    # Initialize learning rate schedulers if requested
    D_scheduler, G_scheduler = None, None
    if lr_decay == 'step':
        D_scheduler = torch.optim.lr_scheduler.StepLR(
            D_solver, step_size=lr_decay_step, gamma=lr_decay_gamma)
        G_scheduler = torch.optim.lr_scheduler.StepLR(
            G_solver, step_size=lr_decay_step, gamma=lr_decay_gamma)
    elif lr_decay == 'plateau':
        D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            D_solver, mode='min', factor=lr_decay_gamma, patience=lr_decay_step, verbose=True)
        G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            G_solver, mode='min', factor=lr_decay_gamma, patience=lr_decay_step, verbose=True)
    elif lr_decay == 'cosine':
        D_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            D_solver, T_max=num_epochs, eta_min=0)
        G_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            G_solver, T_max=num_epochs, eta_min=0)
    
    for epoch in range(num_epochs):
        D.train()
        G.train()
        d_losses = []
        g_losses = []
        
        for batch_idx, (x, y) in enumerate(loader_train):
            if x.size(0) != batch_size:
                continue
                
            # Move data to the appropriate device
            x, y = x.to(device), y.to(device)
            
            # Sample noise
            eta = sample_noise(x.size(0), dim=noise_dim, 
                              distribution=noise_distribution).to(device)
            
            # Prepare inputs
            d_input = torch.cat([x, y.view(batch_size, Ydim)], dim=1)     
            g_input = torch.cat([x, eta], dim=1)
            
            # ==================== Train Discriminator ====================
            D_solver.zero_grad()
            logits_real = D(d_input)
            
            fake_y = G(g_input).detach()
            fake_images = torch.cat([x, fake_y.view(batch_size, Ydim)], dim=1)
                
            logits_fake = D(fake_images)
            
            penalty = calculate_gradient_penalty(D, d_input, fake_images, device)
            d_error = discriminator_loss(logits_real, logits_fake) + 10 * penalty
            d_error.backward()
            D_solver.step()
            d_losses.append(d_error.item())
             
            # ==================== Train Generator ====================
            G_solver.zero_grad()
            
            # First: Standard WGAN loss
            fake_y = G(g_input)
            fake_images = torch.cat([x, fake_y.view(batch_size, Ydim)], dim=1)
            logits_fake = D(fake_images)
            g_error_w = generator_loss(logits_fake)

            # Second: Generate multiple outputs and compute L2 loss against expected y
            if lambda_l>0:  #if lambda_l = 0, then it becomes the standard cWGAN
                g_output = torch.zeros([J_size, batch_size], device=device)
                for i in range(J_size):
                    eta = sample_noise(x.size(0), noise_dim, distribution=noise_distribution).to(device)
                    g_input_i = torch.cat([x, eta], dim=1)
                    g_output[i] = G(g_input_i).view(batch_size)

            # Calculate L2 loss between mean prediction and target
            g_error_l = l2_loss(g_output.mean(dim=0), y.view(batch_size))

            # Combined loss with wasserstein and L2 regularization
            g_error = lambda_w * g_error_w + lambda_l * g_error_l
          
            g_error.backward()
            G_solver.step()
            g_losses.append(g_error.item())
            
            # Increment iteration counter
            iter_count += 1
            
            # Validate and save best model
            if (iter_count >= start_eva) and (iter_count % eva_iter == 0):
                l1_acc, l2_acc = val_G(G=G, loader_data=loader_val, noise_dim=noise_dim, Ydim=Ydim, num_samples=num_samples, device=device,  multivariate=multivariate, reshape_input=False)
                
                print(f"Epoch {epoch}, Iter {iter_count}, "
                      f"D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}, "
                      f"L1: {l1_acc:.4f}, L2: {l2_acc:.4f}")
                
                # Save model if validation improves
                if l2_acc < best_acc:
                    best_acc = l2_acc
                    best_model_g = copy.deepcopy(G.state_dict())
                    best_model_d = copy.deepcopy(D.state_dict())
                    
                    # Save models
                    torch.save(G.state_dict(), f"{save_path}/G_best.pth")
                    torch.save(D.state_dict(), f"{save_path}/D_best.pth")
                    print(f"Saved best model with L2: {best_acc:.4f}")
        
        # Apply learning rate decay at the end of each epoch
        epoch_d_loss = np.mean(d_losses)
        epoch_g_loss = np.mean(g_losses)
        
        print(f"Epoch {epoch} - "
              f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}")
        
        if lr_decay == 'step' or lr_decay == 'cosine':
            if D_scheduler is not None:
                D_scheduler.step()
            if G_scheduler is not None:
                G_scheduler.step()
        elif lr_decay == 'plateau':
            if D_scheduler is not None:
                D_scheduler.step(epoch_d_loss)
            if G_scheduler is not None:
                G_scheduler.step(l2_acc)  # Use validation L2 for generator
        
        # Print current learning rates
        if lr_decay:
            d_lr = D_solver.param_groups[0]['lr']
            g_lr = G_solver.param_groups[0]['lr']
            print(f"Epoch {epoch} - D LR: {d_lr:.6f}, G LR: {g_lr:.6f}")
    
    # Load the best model at the end of training
    G.load_state_dict(best_model_g)
    D.load_state_dict(best_model_d)
    
    return best_acc, G, D
