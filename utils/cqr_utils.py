import torch
import numpy as np
import torch.optim as optim

def quantile_loss(y_true, y_pred, quantile):
    """quantile loss function"""
    error = y_true - y_pred
    return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

def train_quantile_net(model, optimizer, X_train, y_train, alpha=0.1, epochs=200, Ydim=1):
   """
   Training networks to fit (alpha/2) lower bound and (1-alpha/2) upper bound
   
   Args:
       model: Neural network model that outputs 2*Ydim values (lower and upper bounds for each dimension)
       optimizer: Optimizer for training
       X_train: Training features
       y_train: Training targets, shape [n_samples, Ydim]
       alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
       epochs: Number of training epochs
       Ydim: Dimension of the output Y
   """
   input_dim = X_train.shape[1]
   
   X_train_tensor = torch.FloatTensor(X_train)
   
   # Handle the shape of y_train - ensure it is [n_samples, Ydim]
   if len(y_train.shape) == 1:
       # If y_train is one-dimensional, expand to [n_samples, 1]
       y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
   else:
       # If y_train is already multidimensional, convert directly to tensor
       y_train_tensor = torch.FloatTensor(y_train)
   
   # Confirm Ydim is correct
   assert y_train_tensor.shape[1] == Ydim, f"Expected Ydim={Ydim}, got {y_train_tensor.shape[1]}"
   
   for epoch in range(epochs):
       model.train()
       optimizer.zero_grad()
       
       # Forward pass
       outputs = model(X_train_tensor)  # Output shape should be [n_samples, 2*Ydim]
       
       # Ensure output dimensions are correct
       assert outputs.shape[1] == 2*Ydim, f"Model should output 2*Ydim={2*Ydim} values, got {outputs.shape[1]}"
       
       # Separate lower and upper bounds
       lower_bounds = outputs[:, :Ydim]  # First Ydim outputs are lower bounds
       upper_bounds = outputs[:, Ydim:]  # Last Ydim outputs are upper bounds
       
       # Calculate loss for each dimension and sum them
       total_loss = 0
       for dim in range(Ydim):
           # Target values for current dimension
           y_dim_target = y_train_tensor[:, dim:dim+1]
           
           # Lower and upper bounds for current dimension
           lower_pred_dim = lower_bounds[:, dim:dim+1]
           upper_pred_dim = upper_bounds[:, dim:dim+1]
           
           # Calculate loss for current dimension
           loss_lower = quantile_loss(y_dim_target, lower_pred_dim, alpha/2)
           loss_upper = quantile_loss(y_dim_target, upper_pred_dim, 1-alpha/2)
           
           # Accumulate loss
           total_loss += (loss_lower + loss_upper)
       
       # Backward pass
       total_loss.backward()
       optimizer.step()
       
       if (epoch+1) % 20 == 0:
           print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}')
   
   return model

def conformal_calibration(model, X_cal, y_cal, alpha=0.05, Ydim=1):
    """
    Use calibration set to compute conformal correction for multi-dimensional response
    
    Args:
        model: Neural network model that outputs 2*Ydim values
        X_cal: Calibration features
        y_cal: Calibration targets with shape [n_samples, Ydim]
        alpha: Miscoverage level (e.g., 0.05 for 95% coverage)
        Ydim: Dimension of the output Y
        
    Returns:
        corrections: Array of correction factors for each dimension, shape [Ydim]
    """
    model.eval()
    with torch.no_grad():
        X_cal_tensor = torch.FloatTensor(X_cal)
        outputs = model(X_cal_tensor)
        
        # Ensure y_cal has correct shape [n_samples, Ydim]
        if len(y_cal.shape) == 1:
            y_cal = y_cal.reshape(-1, 1)
        
        # Extract lower and upper predictions for each dimension
        lower_pred = outputs[:, :Ydim]  # Shape: [n_samples, Ydim]
        upper_pred = outputs[:, Ydim:]  # Shape: [n_samples, Ydim]
    
    # Calculate nonconformity scores for each dimension
    corrections = []
    
    for dim in range(Ydim):
        # Extract predictions and targets for current dimension
        lower_pred_dim = lower_pred[:, dim]
        upper_pred_dim = upper_pred[:, dim]
        y_cal_dim = y_cal[:, dim]
        
        # Calculate nonconformity scores
        lower_scores = lower_pred_dim - y_cal_dim
        upper_scores = y_cal_dim - upper_pred_dim
        
        # Take maximum of scores
        scores = torch.maximum(lower_scores, upper_scores)
        
        # Find correction factor
        n_cal = len(y_cal)
        adjusted_quantile = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        
        # Calculate the correction (use numpy for quantile computation)
        correction = np.quantile(scores.numpy(), adjusted_quantile)
        corrections.append(correction)
    
    # Return array of corrections
    return np.array(corrections)

def predict_intervals(model, X_test, corrections, Ydim=1):
    """
    Apply correction factors to generate prediction intervals for multi-dimensional response
    
    Args:
        model: Neural network model
        X_test: Test features
        corrections: Array of correction factors from conformal_calibration
        Ydim: Dimension of the output Y
        
    Returns:
        lower_bounds: Lower bounds of prediction intervals, shape [n_samples, Ydim]
        upper_bounds: Upper bounds of prediction intervals, shape [n_samples, Ydim]
    """
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        
        # Extract predictions
        lower_pred = outputs[:, :Ydim]  # Shape: [n_samples, Ydim]
        upper_pred = outputs[:, Ydim:]  # Shape: [n_samples, Ydim]
    
    # Apply conformal correction to each dimension
    lower_bounds = lower_pred.numpy()
    upper_bounds = upper_pred.numpy()
    
    for dim in range(Ydim):
        lower_bounds[:, dim] -= corrections[dim]
        upper_bounds[:, dim] += corrections[dim]
    
    return lower_bounds, upper_bounds
    
def compute_CP(y_test, lower_bounds, upper_bounds, Ydim=1):
   """Compute coverage probability for multi-dimensional prediction intervals."""
   # Convert to numpy arrays if needed
   if isinstance(y_test, torch.Tensor): y_test = y_test.numpy()
   if isinstance(lower_bounds, torch.Tensor): lower_bounds = lower_bounds.numpy()
   if isinstance(upper_bounds, torch.Tensor): upper_bounds = upper_bounds.numpy()
   
   # Ensure correct shapes
   if len(y_test.shape) == 1 and Ydim == 1:
       y_test = y_test.reshape(-1, 1)
   
   # Calculate coverage
   in_interval = (y_test >= lower_bounds) & (y_test <= upper_bounds)
   dim_coverage = np.mean(in_interval, axis=0)
    
   
   # Calculate interval widths
   widths = np.mean(upper_bounds - lower_bounds, axis=0)
   
   print(f"Average coverage: {np.mean(dim_coverage):.4f}")
 
   print(f"Average width: {np.mean(widths):.4f}")
   
   return dim_coverage,  widths    
