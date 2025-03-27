import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

class DatasetLoader:
    def __init__(self, data_root='./data', batch_size=64, num_workers=2, download=True):
        """
        Initialize the dataset loader with parameters
        
        Parameters:
        data_root (str): Directory to store the datasets
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        download (bool): Whether to download the datasets if not present
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        
        # Create data directory if it doesn't exist
        os.makedirs(data_root, exist_ok=True)
        
        # Define transforms for each dataset
        self.mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        self.stl10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
    
    def load_mnist(self):
        """
        Load the MNIST dataset
        
        Returns:
        tuple: (train_loader, test_loader, classes)
        """
        # Download and load training data
        train_dataset = torchvision.datasets.MNIST(
            root=self.data_root,
            train=True,
            transform=self.mnist_transform,
            download=self.download
        )
        
        # Download and load test data
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_root,
            train=False,
            transform=self.mnist_transform,
            download=self.download
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        classes = list(range(10))  # 0-9 digits
        
        print(f"MNIST: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
        return train_loader, test_loader, classes
    
    def load_cifar10(self):
        """
        Load the CIFAR-10 dataset
        
        Returns:
        tuple: (train_loader, test_loader, classes)
        """
        # Download and load training data
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root,
            train=True,
            transform=self.cifar10_transform,
            download=self.download
        )
        
        # Download and load test data
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root,
            train=False,
            transform=self.cifar10_transform,
            download=self.download
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"CIFAR-10: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
        return train_loader, test_loader, classes
    
    def load_stl10(self):
        """
        Load the STL-10 dataset
        
        Returns:
        tuple: (train_loader, test_loader, classes)
        """
        # Download and load training data
        train_dataset = torchvision.datasets.STL10(
            root=self.data_root,
            split='train',
            transform=self.stl10_transform,
            download=self.download
        )
        
        # Download and load test data
        test_dataset = torchvision.datasets.STL10(
            root=self.data_root,
            split='test',
            transform=self.stl10_transform,
            download=self.download
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        classes = ['airplane', 'bird', 'car', 'cat', 'deer', 
                  'dog', 'horse', 'monkey', 'ship', 'truck']
        
        print(f"STL-10: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
        return train_loader, test_loader, classes
    
    def visualize_samples(self, dataset_name, num_samples=10):
        """
        Visualize random samples from the specified dataset
        
        Parameters:
        dataset_name (str): Name of the dataset ('mnist', 'cifar10', or 'stl10')
        num_samples (int): Number of samples to visualize
        """
        if dataset_name.lower() == 'mnist':
            dataset = torchvision.datasets.MNIST(
                root=self.data_root,
                train=True,
                transform=transforms.ToTensor(),
                download=False
            )
            cmap = 'gray'
        elif dataset_name.lower() == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(
                root=self.data_root,
                train=True,
                transform=transforms.ToTensor(),
                download=False
            )
            cmap = None
        elif dataset_name.lower() == 'stl10':
            dataset = torchvision.datasets.STL10(
                root=self.data_root,
                split='train',
                transform=transforms.ToTensor(),
                download=False
            )
            cmap = None
        else:
            raise ValueError("Dataset name must be 'mnist', 'cifar10', or 'stl10'")
        
        # Get random indices
        indices = torch.randperm(len(dataset))[:num_samples]
        
        # Plot samples
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
        
        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            img = img.numpy().transpose((1, 2, 0))
            
            # For MNIST, remove the channel dimension (1, 28, 28) -> (28, 28)
            if dataset_name.lower() == 'mnist':
                img = img.squeeze()
            
            axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(f"Class: {label}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Initialize dataset loader
    loader = DatasetLoader(batch_size=32)
    
    # Load MNIST dataset
    mnist_train_loader, mnist_test_loader, mnist_classes = loader.load_mnist()
    
    # Load CIFAR-10 dataset
    cifar10_train_loader, cifar10_test_loader, cifar10_classes = loader.load_cifar10()
    
    # Load STL-10 dataset
    stl10_train_loader, stl10_test_loader, stl10_classes = loader.load_stl10()
    
    # Visualize samples from each dataset
    loader.visualize_samples('mnist', num_samples=5)
    loader.visualize_samples('cifar10', num_samples=5)
    loader.visualize_samples('stl10', num_samples=5)
    
    # Test data iteration
    for images, labels in mnist_train_loader:
        print(f"MNIST batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
        
    for images, labels in cifar10_train_loader:
        print(f"CIFAR-10 batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
        
    for images, labels in stl10_train_loader:
        print(f"STL-10 batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
