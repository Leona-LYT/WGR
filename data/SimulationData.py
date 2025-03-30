import torch
import torch.distributions as dist
from multivariateY import generate_Gaussian_mixture_data, generate_involute_data, generate_octagon_data

# =============================================================================
# Data generation models
# =============================================================================
class DataGenerator:
    """Data generator for various simulation models (M1, M2, M3, M4, SM1, SM2, SM3, SM4)."""
    
    def __init__(self, args):
        """Initialize with args containing train, val, test sizes and Xdim."""
        self.args = args
        self.t_dist = dist.StudentT(df=3.0)
        self.beta = torch.tensor([1, 1, -1, -1, 1] + [0]*(args.Xdim-5)).float()
        
        # Model generators mapping
        self.model_generators = {
            'M1': self._generate_M1_data,
            'M2': self._generate_M2_data,
            'M3': self._generate_M3_data,
            'M4': self._generate_M4_data,
            'SM1': self._generate_SM1_data,
            'SM2': self._generate_SM2_data,
            'SM3': self._generate_SM3_data,
            'SM4': self._generate_SM4_data
        }
        
    def generate_data(self, model_name):
        """Generate data for specified model, returns dict with train/val/test data."""
        if model_name not in self.model_generators:
            raise ValueError(f"Unknown model: {model_name}. Supported: {list(self.model_generators.keys())}")
        return self.model_generators[model_name]()
    
    def _generate_dataset(self, X_generator, Y_generator):
        """Generate datasets using provided generator functions."""
        datasets = {}
        for prefix, size in [('train', self.args.train), ('val', self.args.val), ('test', self.args.test)]:
            X = torch.randn([size, self.args.Xdim])
            eps = X_generator(size)
            Y = Y_generator(X, eps)
            datasets[f'{prefix}_X'] = X
            datasets[f'{prefix}_Y'] = Y
        return datasets
        
    def _split_datasets(self, X, Y):
        """Split generated datasets into train/val/test sets."""
        train_idx, val_idx = self.args.train, self.args.train + self.args.val
        
        return {
            'train_X': X[:train_idx], 'train_Y': Y[:train_idx],
            'val_X': X[train_idx:val_idx], 'val_Y': Y[train_idx:val_idx],
            'test_X': X[val_idx:], 'test_Y': Y[val_idx:]
        }
        
    def _generate_M1_data(self):
        """M1 model data generator."""
        def gen_eps(size):
            return torch.randn([size])
        
        def gen_Y(X, eps):
            return (X[:,0]**2 + torch.exp(X[:,1]+X[:,2]/3) + X[:,3] - X[:,4] + 
                   (0.5 + X[:,1]**2/2 + X[:,4]**2/2)*eps)
        
        return self._generate_dataset(gen_eps, gen_Y)
    
    def _generate_M2_data(self):
        """M2 model data generator."""
        def gen_eps(size):
            return torch.randn([size])
        
        def gen_Y(X, eps):
            SI = X @ self.beta
            return SI**2 + torch.sin(torch.abs(SI)) + 2*torch.cos(eps)
        
        return self._generate_dataset(gen_eps, gen_Y)
        
    def _generate_M3_data(self): 
        """Generate data for M3 model (independ Gaussian mixture)"""
        X, Y = generate_Gaussian_mixture_data(n_samples=self.args.train + self.args.val + self.args.test)
    
        return self._split_datasets(X, Y)
        
    def _generate_M4_data(self): 
        """Generate data for M4 model (involute mixture)"""
        X, Y = generate_involute_data(n_samples=self.args.train + self.args.val + self.args.test)
    
        return self._split_datasets(X, Y)
        
    def _generate_SM1_data(self):
        """SM1 model data generator."""
        def gen_eps(size):
            return torch.randn([size])
        
        def gen_Y(X, eps):
            return X[:,0]**2 + torch.exp(X[:,1]+X[:,2]/3) + torch.sin(X[:,3]+X[:,4]) + eps
        
        return self._generate_dataset(gen_eps, gen_Y)
    
    def _generate_SM2_data(self):
        """SM2 model data generator."""
        def gen_eps(size):
            return self.t_dist.sample(sample_shape=([size]))
        
        def gen_Y(X, eps):
            return X[:,0]**2 + torch.exp(X[:,1] + X[:,2]/3) + X[:,3] - X[:,4] + eps
        
        return self._generate_dataset(gen_eps, gen_Y)
    
    def _generate_SM3_data(self):
        """SM3 model data generator."""
        def gen_eps(size):
            return 0.5*(torch.randn([size])-2) + 0.5*(torch.randn([size])+2)
        
        def gen_Y(X, eps):
            return (5 + X[:,0]**2/3 + X[:,1]**2 + X[:,2]**2 + X[:,3] + X[:,4]) * torch.exp(0.5*eps)
        
        return self._generate_dataset(gen_eps, gen_Y)
    
    def _generate_SM4_data(self):
        """Generate data for SM4 model (octagon)"""
        X, Y = generate_octagon_data(n_samples=self.args.train + self.args.val + self.args.test)
    
        return self._split_datasets(X, Y)
        
# =============================================================================
# data generator for visualization
# =============================================================================
def generate_multi_responses_multiY(x_value, n_responses=100, model_type="gaussian_mixture"):
    """
    Generate multiple Y responses for a single X value using different data generation functions.
    
    Parameters:
    x_value (float): The X value to generate responses for
    n_responses (int): Number of different responses to generate
    model_type (str): Type of data to generate - "M3"(gaussian_mixture), "M4"(involute), or "SM4"(octagon)
    
    Returns:
    Y: Y contains n_responses 2D coordinates
    """
    # Create a tensor with the x_value repeated n_responses times
    X = torch.full((n_responses,), x_value)
    
    # Select the appropriate data generation function
    if model_type == "M3":
        _, Y = generate_Gaussian_mixture_data(X=X, n_samples=n_responses)
    elif model_type == "M4":
        _, Y = generate_involute_data(X=X, n_samples=n_responses)
    elif model_type == "SM4":
        _, Y = generate_octagon_data(X=X, n_samples=n_responses)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'M3', 'M4', or 'SM4'.")
    
    return Y
