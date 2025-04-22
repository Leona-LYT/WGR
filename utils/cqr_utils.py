import torch

def quantile_loss(y_true, y_pred, quantile):
    """quantile loss function"""
    error = y_true - y_pred
    return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

def train_quantile_net(X_train, y_train, alpha=0.1, epochs=200, lr=0.001):
    """training networks to fit (alpha/2) lower bound and (1-alpha/2) upper bound"""
    input_dim = X_train.shape[1]
    model = QuantileNet(input_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # feedforward 
        outputs = model(X_train_tensor)
        lower_pred = outputs[:, 0].unsqueeze(1)
        upper_pred = outputs[:, 1].unsqueeze(1)
        
        # Calculate losses
        loss_lower = quantile_loss(y_train_tensor, lower_pred, alpha/2)
        loss_upper = quantile_loss(y_train_tensor, upper_pred, 1-alpha/2)
        loss = loss_lower + loss_upper
        
        # backward 
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def conformal_calibration(model, X_cal, y_cal, alpha=0.05):
    """Use calibration set to compute conformal correction"""
    model.eval()
    with torch.no_grad():
        X_cal_tensor = torch.FloatTensor(X_cal)
        outputs = model(X_cal_tensor)
        lower_pred = outputs[:, 0] 
        upper_pred = outputs[:, 1] 
    
    # Calculate nonconformity scores
    lower_scores = lower_pred - y_cal
    upper_scores = y_cal - upper_pred
    
    # Find correction factors
    n_cal = len(y_cal)
    adjusted_quantile = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    
    lower_correction = np.quantile(lower_scores, adjusted_quantile)
    upper_correction = np.quantile(upper_scores, adjusted_quantile)
    
    return lower_correction, upper_correction

def predict_intervals(model, X_test, lower_correction, upper_correction):
    """Apply correction factors to generate prediction intervals"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        lower_pred = outputs[:, 0] 
        upper_pred = outputs[:, 1] 
    
    # Apply conformal correction
    lower_bounds = lower_pred - lower_correction
    upper_bounds = upper_pred + upper_correction
    
    return lower_bounds, upper_bounds
