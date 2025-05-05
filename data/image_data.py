import os
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

# =============================================================================
# Create custom datasets that include X, Y, XY and labels(classes)
# =============================================================================
class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, XY, labels):
        self.X = X
        self.Y = Y
        self.XY = XY
        self.labels = labels
            
    def __len__(self):
        return len(self.X)
            
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.XY[idx], self.labels[idx]
        
class DatasetLoader:
    def __init__(self, data_root='./data', num_workers=2, download=True):
        """
        Initialize the dataset loader with parameters
        
        Parameters:
        data_root (str): Directory to store the datasets
        num_workers (int): Number of workers for data loading
        download (bool): Whether to download the datasets if not present
        """
        self.data_root = data_root
        self.num_workers = num_workers
        self.download = download
        
        # Create data directory if it doesn't exist
        os.makedirs(data_root, exist_ok=True)
        
        # Define transforms for each dataset
        self.mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        self.stl10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
    
    def load_mnist(self, train_size, val_size, test_size, train_batch, val_batch, test_batch,  mask_size=14, mask_position=(7, 7)):
        """
        Load the MNIST dataset with options to select subsets

        Parameters:
        train_size (int): Number of samples for training 
        val_size (int): Number of samples for validation 
        test_size (int): Number of samples for testing 

        train_batch (int): Batch size for train data 
        val_batch (int): Batch size for validation data
        test_batch (int): Batch size for testing data

        apply_mask (bool): Whether to apply a mask to the center of the images
        mask_size (int): Size of the square mask to apply
        mask_position (tuple): Top-left position (x, y) to start the mask
        
        Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
        """
        
        # Download and load training data
        full_train_dataset = torchvision.datasets.MNIST( root=self.data_root, train=True, transform=self.mnist_transform, download=self.download )
        
        # Download and load test data
        full_test_dataset = torchvision.datasets.MNIST( root=self.data_root, train=False, transform=self.mnist_transform, download=self.download )

        # Extract images and labels
        train_images = []
        train_labels = []
        for img, label in full_train_dataset:
            train_images.append(img)
            train_labels.append(label)

        train_images = torch.stack(train_images)
        train_labels = torch.tensor(train_labels)

        test_images = []
        test_labels = []
        for img, label in full_test_dataset:
            test_images.append(img)
            test_labels.append(label)
        
        test_images = torch.stack(test_images)
        test_labels = torch.tensor(test_labels)
        
        total_train = len(train_images)
        train_size = min(train_size, total_train)
        val_size = min(val_size, total_train - train_size)
        
        # Split data
        X_train_orig = train_images[:train_size]
        labels_train = train_labels[:train_size]
        
        X_val_orig = train_images[train_size:train_size + val_size]
        labels_val = train_labels[train_size:train_size + val_size]

        X_test_orig = test_images[:test_size]
        labels_test = test_labels[:test_size]

        # mask samples
        X_train, Y_train = self._apply_mask(X_train_orig, mask_size, mask_position)
        X_val, Y_val = self._apply_mask(X_val_orig, mask_size, mask_position)
        X_test, Y_test = self._apply_mask(X_test_orig, mask_size, mask_position)

        # Create datasets
        train_dataset = MaskedDataset(X_train, Y_train, X_train_orig, labels_train)
        test_dataset = MaskedDataset(X_test, Y_test, X_test_orig, labels_test)
        val_dataset = MaskedDataset(X_val, Y_val, X_val_orig, labels_val)
 
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=self.num_workers)
        
        val_loader = DataLoader( val_dataset, batch_size=val_batch, shuffle=False, num_workers=self.num_workers )

        test_loader = DataLoader( test_dataset, batch_size=test_batch, shuffle=False, num_workers=self.num_workers )

        classes = list(range(10))
        
        print(f"Applied {mask_size}x{mask_size} mask at position {mask_position}")

        return train_loader, val_loader, test_loader, classes
        
    def load_cifar10(self, train_size, val_size, test_size, train_batch, val_batch, test_batch,  mask_size=14, mask_position=(7, 7)):
        """
        Load the CIFAR-10 dataset with options to select subsets

        Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
        """
        
        # Download and load training data
        full_train_dataset = torchvision.datasets.CIFAR10( root=self.data_root, train=True, transform=self.cifar10_transform, download=self.download )
        
        # Download and load test data
        full_test_dataset = torchvision.datasets.CIFAR10( root=self.data_root, train=False, transform=self.cifar10_transform, download=self.download )

        # Extract images and labels
        train_images = []
        train_labels = []
        for img, label in full_train_dataset:
            train_images.append(img)
            train_labels.append(label)

        train_images = torch.stack(train_images)
        train_labels = torch.tensor(train_labels)

        test_images = []
        test_labels = []
        for img, label in full_test_dataset:
            test_images.append(img)
            test_labels.append(label)
        
        test_images = torch.stack(test_images)
        test_labels = torch.tensor(test_labels)
        
        total_train = len(train_images)
        train_size = min(train_size, total_train)
        val_size = min(val_size, total_train - train_size)
        
        # Split data
        X_train_orig = train_images[:train_size]
        labels_train = train_labels[:train_size]
        
        X_val_orig = train_images[train_size:train_size + val_size]
        labels_val = train_labels[train_size:train_size + val_size]

        X_test_orig = test_images[:test_size]
        labels_test = test_labels[:test_size]

        # mask samples
        X_train, Y_train = self._apply_mask(X_train_orig, mask_size, mask_position)
        X_val, Y_val = self._apply_mask(X_val_orig, mask_size, mask_position)
        X_test, Y_test = self._apply_mask(X_test_orig, mask_size, mask_position)

        # Create datasets
        train_dataset = MaskedDataset(X_train, Y_train, X_train_orig, labels_train)
        test_dataset = MaskedDataset(X_test, Y_test, X_test_orig, labels_test)
        val_dataset = MaskedDataset(X_val, Y_val, X_val_orig, labels_val)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=self.num_workers)
        
        val_loader = DataLoader( val_dataset, batch_size=val_batch, shuffle=False, num_workers=self.num_workers )

        test_loader = DataLoader( test_dataset, batch_size=test_batch, shuffle=False, num_workers=self.num_workers )

        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"Applied {mask_size}x{mask_size} mask at position {mask_position}")

        return train_loader, val_loader, test_loader, classes    

    def load_stl10(self, train_size, val_size, test_size, train_batch, val_batch, test_batch,  mask_size=14, mask_position=(7, 7)):
        """
        Load the STL-10 dataset with options to select subsets

        Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
        """
        
        # Download and load training data
        full_train_dataset = torchvision.datasets.STL10( root=self.data_root, train=True, transform=self.STL10_transform, download=self.download )
        
        # Download and load test data
        full_test_dataset = torchvision.datasets.STL10( root=self.data_root, train=False, transform=self.STL10_transform, download=self.download )

        # Extract images and labels
        train_images = []
        train_labels = []
        for img, label in full_train_dataset:
            train_images.append(img)
            train_labels.append(label)

        train_images = torch.stack(train_images)
        train_labels = torch.tensor(train_labels)

        test_images = []
        test_labels = []
        for img, label in full_test_dataset:
            test_images.append(img)
            test_labels.append(label)
        
        test_images = torch.stack(test_images)
        test_labels = torch.tensor(test_labels)
        
        total_train = len(train_images)
        train_size = min(train_size, total_train)
        val_size = min(val_size, total_train - train_size)
        
        # Split data
        X_train_orig = train_images[:train_size]
        labels_train = train_labels[:train_size]
        
        X_val_orig = train_images[train_size:train_size + val_size]
        labels_val = train_labels[train_size:train_size + val_size]

        X_test_orig = test_images[:test_size]
        labels_test = test_labels[:test_size]

        # mask samples
        X_train, Y_train = self._apply_mask(X_train_orig, mask_size, mask_position)
        X_val, Y_val = self._apply_mask(X_val_orig, mask_size, mask_position)
        X_test, Y_test = self._apply_mask(X_test_orig, mask_size, mask_position)

        # Create datasets
        train_dataset = MaskedDataset(X_train, Y_train, X_train_orig, labels_train)
        test_dataset = MaskedDataset(X_test, Y_test, X_test_orig, labels_test)
        val_dataset = MaskedDataset(X_val, Y_val, X_val_orig, labels_val)
 
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=self.num_workers)
        
        val_loader = DataLoader( val_dataset, batch_size=val_batch, shuffle=False, num_workers=self.num_workers )

        test_loader = DataLoader( test_dataset, batch_size=test_batch, shuffle=False, num_workers=self.num_workers )

        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"Applied {mask_size}x{mask_size} mask at position {mask_position}")

        return train_loader, val_loader, test_loader, classes  
        
    def _apply_mask(self, images, mask_size=12, mask_position=(7, 7)):
        """
        Apply a mask to images and return both masked images and the masked regions.
        
        Parameters:
        images (torch.Tensor): Tensor of images
        mask_size (int): Size of the square mask
        mask_position (tuple): Top-left position to start the mask
        
        Returns:
        tuple: (masked_images, masked_regions)
        """
        # Get basic parameters
        num_images = images.shape[0]
        image_shape = images.shape[1:]  # Could be [1, 28, 28] for MNIST
        
        # Create copies to avoid modifying originals
        masked_images = images.clone()
        
        # Initialize tensor to hold masked regions
        x, y = mask_position
        masked_regions = torch.zeros(num_images, 1, mask_size, mask_size)
        
        # Apply mask to each image
        for i in range(num_images):
            # Extract the region to be masked
            if len(image_shape) == 3:  # [channels, height, width]
                masked_regions[i, 0] = images[i, 0, y:y+mask_size, x:x+mask_size]
                # Apply mask (set to zero)
                masked_images[i, 0, y:y+mask_size, x:x+mask_size] = 0
            else:  # Handle other shapes if needed
                raise ValueError(f"Unexpected image shape: {image_shape}")
        
        return masked_images, masked_regions
    
    def visualize_samples(self, dataset_name, num_samples=10, apply_mask=False, mask_size=12, mask_position=(7, 7)):
        """
        Visualize random samples from the specified dataset
        
        Parameters:
        dataset_name (str): Name of the dataset ('mnist', 'cifar10', or 'stl10')
        num_samples (int): Number of samples to visualize
        apply_mask (bool): Whether to show masked images
        mask_size (int): Size of the square mask
        mask_position (tuple): Top-left position to start the mask
        """
        transform = transforms.ToTensor()
        if dataset_name.lower() == 'mnist':
            dataset = torchvision.datasets.MNIST(
                root=self.data_root,
                train=True,
                transform=transform,
                download=False
            )
            cmap = 'gray'
            classes = list(range(10))
        elif dataset_name.lower() == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(
                root=self.data_root,
                train=True,
                transform=transform,
                download=False
            )
            cmap = None
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                     'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset_name.lower() == 'stl10':
            dataset = torchvision.datasets.STL10(
                root=self.data_root,
                split='train',
                transform=transform,
                download=False
            )
            cmap = None
            classes = ['airplane', 'bird', 'car', 'cat', 'deer', 
                     'dog', 'horse', 'monkey', 'ship', 'truck']
        else:
            raise ValueError("Dataset name must be 'mnist', 'cifar10', or 'stl10'")
        
        # Get random indices
        total_samples = len(dataset)
        if num_samples > total_samples:
            num_samples = total_samples
            print(f"Warning: Requested more samples than available. Showing {num_samples} samples.")
            
        indices = torch.randperm(total_samples)[:num_samples]
        
        # Collect images
        images = []
        labels = []
        for idx in indices:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)
        
        images = torch.stack(images)
        
        # Apply mask if requested
        if apply_mask:
            masked_images, masked_regions = self._apply_mask(images, mask_size, mask_position)
            
            # Create figure with two rows: masked images and masked regions
            fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
            
            for i in range(num_samples):
                # Display masked image
                img = masked_images[i].numpy().transpose((1, 2, 0))
                if dataset_name.lower() == 'mnist':
                    img = img.squeeze()
                axes[0, i].imshow(img, cmap=cmap)
                axes[0, i].set_title(f"Class: {classes[labels[i]]}")
                axes[0, i].axis('off')
                
                # Display masked region
                mask_img = masked_regions[i].numpy().transpose((1, 2, 0))
                if dataset_name.lower() == 'mnist':
                    mask_img = mask_img.squeeze()
                axes[1, i].imshow(mask_img, cmap=cmap)
                axes[1, i].set_title("Masked Region")
                axes[1, i].axis('off')
        else:
            # Just display regular images
            fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
            if num_samples == 1:
                axes = [axes]  # Make axes iterable when there's only one sample
                
            for i in range(num_samples):
                img = images[i].numpy().transpose((1, 2, 0))
                if dataset_name.lower() == 'mnist':
                    img = img.squeeze()
                axes[i].imshow(img, cmap=cmap)
                axes[i].set_title(f"Class: {classes[labels[i]]}")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def show_images(self, images, num_images=None, title=None):
        """
        Display a batch of images
        
        Parameters:
        images (torch.Tensor): Tensor of images to display
        num_images (int): Number of images to show, defaults to all
        title (str): Optional title for the figure
        """
        # Determine number of images to show
        if num_images is None:
            num_images = min(5, len(images))
        else:
            num_images = min(num_images, len(images))
        
        # Create figure
        fig, axes = plt.subplots(1, num_images, figsize=(2*num_images, 2))
        if num_images == 1:
            axes = [axes]  # Make axes iterable when there's only one image
            
        # Convert to numpy and display each image
        for i in range(num_images):
            img = images[i].detach().cpu()
            if len(img.shape) == 3:  # [channels, height, width]
                img = img.numpy().transpose((1, 2, 0))
                if img.shape[2] == 1:  # Single channel - use grayscale
                    img = img.squeeze()
                    cmap = 'gray'
                else:
                    cmap = None
            else:  # Already 2D
                img = img.numpy()
                cmap = 'gray'
                
            axes[i].imshow(img, cmap=cmap)
            axes[i].axis('off')
            
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        plt.show()
        return fig

