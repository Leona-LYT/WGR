import torch
import torch.distributions as dist

# =============================================================================
# Univerate Y
# =============================================================================
class DataGenerator_UniY:
    """Data generator for various simulation models (M1, M2, SM1, SM2, SM3)."""
    
    def __init__(self, args):
        """Initialize with args containing train, val, test sizes and Xdim."""
        self.args = args
        self.t_dist = dist.StudentT(df=3.0)
        self.beta = torch.tensor([1, 1, -1, -1, 1] + [0]*(args.Xdim-5)).float()
        
        # Model generators mapping
        self.model_generators = {
            'M1': self._generate_M1_data,
            'M2': self._generate_M2_data,
            'SM1': self._generate_SM1_data,
            'SM2': self._generate_SM2_data,
            'SM3': self._generate_SM3_data
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
    
    # Split into train/val/test sets
    train_X = X[:self.args.train]
    val_X = X[self.args.train:self.args.train + self.args.val]
    test_X = X[self.args.train + self.args.val:]
    
    train_Y = Y[:self.args.train]
    val_Y = Y[self.args.train:self.args.train + self.args.val]
    test_Y = Y[self.args.train + self.args.val:]
    
    return {
        'train_X': train_X, 'train_Y': train_Y,
        'val_X': val_X, 'val_Y': val_Y,
        'test_X': test_X, 'test_Y': test_Y
    }
# =============================================================================
# Multiverate Y
# =============================================================================
import torch
import numpy as np
import math

def generate_octagon_data(n_samples=17000):
    """
    Generate data points forming an octagon pattern.
    
    Parameters:
    n_samples (int): Number of samples to generate
    
    Returns:
    tuple: (X, Y) where X is the input and Y contains the 2D coordinates
    """
    # Generate one-dimensional X
    X = torch.from_numpy(np.random.normal(0, 1, n_samples))
    
    # Create interval indicator function
    def create_interval_indicators(u_values, n_intervals=8):
        """Create indicators for which interval each u value falls into"""
        # Convert to tensor for easier operations
        u_tensor = torch.tensor(u_values, dtype=torch.float32)
        
        # Initialize result tensor
        indicators = torch.zeros((n_samples, n_intervals))
        
        # Set indicators for each interval
        for i in range(n_intervals):
            indicators[:, i] = ((u_tensor > i) & (u_tensor < (i+1))).float()
            
        return indicators
    
    # Generate uniform samples
    U1 = np.random.uniform(0, 8, n_samples)
    U2 = np.random.uniform(0, 8, n_samples)
    
    # Get indicators for which interval each sample falls into
    U1_ind = create_interval_indicators(U1)
    U2_ind = create_interval_indicators(U2)
    
    # Calculate means for each octagon vertex
    angles = torch.tensor(np.arange(1, 9) * math.pi / 4)
    mu1 = 3 * torch.cos(angles)
    mu2 = 3 * torch.sin(angles)
    Mu = torch.stack([mu1, mu2], dim=1)
    
    # Calculate covariance matrices for each vertex
    s11 = np.cos(angles)**2 + 0.16**2 * np.sin(angles)**2
    s12 = s21 = (1 - 0.16**2) * np.cos(angles) * np.sin(angles)
    s22 = np.sin(angles)**2 + 0.16**2 * np.cos(angles)**2
    
    # Initialize terms for the sum
    eps1_term = torch.zeros(n_samples, 8)
    eps2_term = torch.zeros(n_samples, 8)
    
    # Generate multivariate normal samples for each vertex
    for i in range(8):
        cov_matrix = np.array([[s11[i], s12[i]], [s21[i], s22[i]]])
        multi_norm_i = torch.from_numpy(
            np.random.multivariate_normal(Mu[i], cov_matrix, n_samples)
        )
        
        # Multiply by indicators
        eps1_term[:, i] = U1_ind[:, i] * multi_norm_i[:, 0]
        eps2_term[:, i] = U2_ind[:, i] * multi_norm_i[:, 1]
    
    # Sum contributions from each vertex
    eps1 = eps1_term.sum(dim=1)
    eps2 = eps2_term.sum(dim=1)
    
    # Create final Y values
    Y1 = X + eps1
    Y2 = X + eps2
    Y = torch.stack([Y1, Y2], dim=1)
    
    return X, Y

# Example usage
X, Y = generate_octagon_data(17000)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# =============================================================================
# Multiverate Y
# =============================================================================
# generate M3 (independ gussian mixture)
# def I1(num):
    if num < 1/3: 
        return 1
    else:
        return 0
    
def I2(num):
    if num > 1/3 and num<2/3: 
        return 1
    else:
        return 0    
    
def I3(num):
    if num > 2/3: 
        return 1
    else:
        return 0  
        
X = torch.from_numpy(np.random.normal(0, 1, 17000))  # one dimensional x
U1 = np.random.uniform(0, 1, 17000)
U2 = np.random.uniform(0, 1, 17000)

I1U1 = torch.Tensor([I1(x) for x in U1])
I2U1 = torch.Tensor([I2(x) for x in U1])
I3U1 = torch.Tensor([I3(x) for x in U1])

I1U2 = torch.Tensor([I1(x) for x in U2])
I2U2 = torch.Tensor([I2(x) for x in U2])
I3U2 = torch.Tensor([I3(x) for x in U2])

eps1 =  I1U1 * np.random.normal(-2, 0.25, 17000) + I2U1 * np.random.normal(0, 0.25, 17000) + I3U1 * np.random.normal(2, 0.25, 17000)
eps2 =  I1U2 * np.random.normal(-2, 0.25, 17000) + I2U2 * np.random.normal(0, 0.25, 17000) + I3U2 * np.random.normal(2, 0.25, 17000) 

Y1 = X + eps1
Y2 = X + eps2

Y = torch.cat([Y1.view(17000,1) ,Y2.view(17000,1) ],dim=1)

# generate M4 (involute)
X = np.random.normal(0, 1, 17000)  # one dimensional x
U = np.random.uniform(0, 2*math.pi, 17000)
eps1 = np.random.normal(0, 0.4, 17000)
eps2 = np.random.normal(0, 0.4, 17000)
Y1 = torch.from_numpy(X + U * np.sin(2*U) + eps1).view(17000,1)  #conditional on x=1 
Y2 = torch.from_numpy(X + U * np.cos(2*U) + eps2).view(17000,1)
Y = torch.cat([Y1,Y2],dim=1)

#generate SM4 (octagon)
def I1(num):
    if num < 1: 
        return 1
    else:
        return 0
    
def I2(num):
    if num > 1 and num<2: 
        return 1
    else:
        return 0    
    
def I3(num):
    if num > 2 and num<3: 
        return 1
    else:
        return 0 
    
def I4(num):
    if num > 3 and num<4: 
        return 1
    else:
        return 0     
    
def I5(num):
    if num > 4 and num<5: 
        return 1
    else:
        return 0     

def I6(num):
    if num > 5 and num<6: 
        return 1
    else:
        return 0     
    
def I7(num):
    if num > 6 and num<7: 
        return 1
    else:
        return 0     
    
def I8(num):
    if num > 7 and num<8: 
        return 1
    else:
        return 0         

X = torch.from_numpy(np.random.normal(0, 1, 17000))  # one dimensional x
U1 = np.random.uniform(0, 8, 17000)
U2 = np.random.uniform(0, 8, 17000)

I1U1 = torch.Tensor([I1(x) for x in U1]).view(17000,1)
I2U1 = torch.Tensor([I2(x) for x in U1]).view(17000,1)
I3U1 = torch.Tensor([I3(x) for x in U1]).view(17000,1)
I4U1 = torch.Tensor([I4(x) for x in U1]).view(17000,1)
I5U1 = torch.Tensor([I5(x) for x in U1]).view(17000,1)
I6U1 = torch.Tensor([I6(x) for x in U1]).view(17000,1)
I7U1 = torch.Tensor([I7(x) for x in U1]).view(17000,1)
I8U1 = torch.Tensor([I8(x) for x in U1]).view(17000,1)

I1U2 = torch.Tensor([I1(x) for x in U2]).view(17000,1)
I2U2 = torch.Tensor([I2(x) for x in U2]).view(17000,1)
I3U2 = torch.Tensor([I3(x) for x in U2]).view(17000,1)
I4U2 = torch.Tensor([I4(x) for x in U2]).view(17000,1)
I5U2 = torch.Tensor([I5(x) for x in U2]).view(17000,1)
I6U2 = torch.Tensor([I6(x) for x in U2]).view(17000,1)
I7U2 = torch.Tensor([I7(x) for x in U2]).view(17000,1)
I8U2 = torch.Tensor([I8(x) for x in U2]).view(17000,1)

U1_ind = torch.cat([I1U1, I2U1, I3U1, I4U1, I5U1, I6U1, I7U1, I8U1],dim=1)
U2_ind = torch.cat([I1U2, I2U2, I3U2, I4U2, I5U2, I6U2, I7U2, I8U2],dim=1)

mu1 = torch.from_numpy(3*np.cos(math.pi * np.arange(1,9)/4)).view(8,1)
mu2 = torch.from_numpy(3*np.sin(math.pi * np.arange(1,9)/4)).view(8,1) 

Mu = torch.cat([mu1,mu2],dim=1)

s11 = np.cos(math.pi * np.arange(1,9)/4)**2 + 0.16**2 * np.sin(math.pi * np.arange(1,9)/4)**2
s12 = s21 = (1- 0.16**2 ) * np.cos(math.pi * np.arange(1,9)/4) * np.sin(math.pi * np.arange(1,9)/4)
s22 = np.sin(math.pi * np.arange(1,9)/4)**2 + 0.16**2 * np.cos(math.pi * np.arange(1,9)/4)**2

eps1_term = torch.zeros([17000,8]) 
eps2_term = torch.zeros([17000,8]) 

multi_norm = torch.zeros([17000,8,2]) 

for i in range(8):
    multi_norm[:,i] = torch.from_numpy(np.random.multivariate_normal(Mu[i], np.array([[s11[i],s12[i]],[s21[i],s22[i]]]), 17000))
      
for i in range(8):   
    eps1_term[:,i] = U1_ind[:,i] * multi_norm[:,i][:,0] 
    eps2_term[:,i] = U2_ind[:,i] * multi_norm[:,i][:,1] 

eps1 = eps1_term.sum(dim=1)
eps2 = eps2_term.sum(dim=1)

Y1 = X + eps1
Y2 = X + eps2

Y = torch.cat([Y1.view(17000,1) ,Y2.view(17000,1) ],dim=1)
