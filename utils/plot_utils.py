import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image 
import seaborn as sns

import torch
import torchvision.transforms as transforms

def convert_generated_to_mnist_range(generated_img):
    """
    Convert a generated image (typically in the range [-1, 1]) to match 
    the pixel intensity range observed in the MNIST dataset.

    Parameters:
        generated_img (Tensor or ndarray): An image tensor with values in [-1, 1].

    Returns:
        converted (Tensor or ndarray): The image rescaled to the MNIST pixel range.
    """
    min_val, max_val = -0.4242, 2.8215
    range_val = max_val - min_val
    
    normalized = (generated_img + 1) / 2
    
    converted = normalized * range_val + min_val
    return converted

def plot_kde_2d(data, cmap='Blues', fill=True, show_cbar=True, thresh=0.05, 
               bw_adjust=0.2, levels=6, fig_size=(8, 6), title=None,xlabel='Y1', ylabel='Y2'):
    """
    Create a 2D kernel density estimation plot for simulated multivariate response data (M4,SM3,SM4).
    
    Parameters:
    data (array-like): Input data with shape (n_samples, 2)
    cmap (str): Colormap for the density plot
    fill (bool): Whether to fill the contour
    show_cbar (bool): Whether to show the color bar
    thresh (float): Threshold for the contour plot
    bw_adjust (float): Bandwidth adjustment factor
    levels (int): Number of contour levels
    fig_size (tuple): Figure size (width, height) in inches
    title (str): Optional title for the plot
    xlabel (str): Label for the x-axis
    ylabel (str): Label for the y-axis
    """
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create the KDE plot
    sns.kdeplot(
        x=data[:, 0], 
        y=data[:, 1],
        cmap=cmap,
        fill=fill,
        cbar=show_cbar,
        thresh=thresh, 
        bw_adjust=bw_adjust, 
        levels=levels,
        common_grid=True,
        cbar_kws={'format': '%.3f'},
        ax=ax
    )
    
    # Add title if provided
    if title:
        ax.set_title(title)
        
    # Add axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Return the figure and axis objects for further customization if needed
    return fig, ax


def CI_Plot(x, y1, y2, y3, title=None, x_label='Sample number', y_label='Location', 
           colors=None, linestyles=None, markers=None, legend_labels=None, legend_loc='lower right',
           figsize=(10, 6), save_path=None, fill_between=False, alpha=0.2):
    """
    Plot prediction intervals and true values
    
    Parameters:
        x: x-axis data (sample numbers)
        y1: true values
        y2: lower bound (e.g., 2.5% quantile)
        y3: upper bound (e.g., 97.5% quantile)
        title: plot title, default is None
        x_label: x-axis label
        y_label: y-axis label
        colors: list of line colors, default is None (use seaborn default colors)
        linestyles: list of line styles, default is None (use seaborn default styles)
        markers: list of marker styles, default is None (use seaborn default markers)
        legend_labels: list of legend labels, default is ['Truth', '2.5% Quantile', '97.5% Quantile']
        legend_loc: legend position
        figsize: figure size, default is (10, 6)
        save_path: path to save the image, default is None (don't save)
    """
    # Set default values
    if legend_labels is None:
        legend_labels = ['Truth', '2.5% Quantile', '97.5% Quantile']
    
    # Ensure input data are numpy arrays
    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    y3 = np.asarray(y3)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Set style
    sns.set(style='white', palette='muted')
    # Default method: plot each line separately
    sns.lineplot(x=x, y=y1, 
                color=colors[0] if colors is not None else None,
                linestyle=linestyles[0] if linestyles is not None else None,
                marker=markers[0] if markers is not None else None,
                label=legend_labels[0])
        
    sns.lineplot(x=x, y=y2, 
                color=colors[1] if colors is not None and len(colors) > 1 else None,
                linestyle=linestyles[1] if linestyles is not None and len(linestyles) > 1 else None,
                marker=markers[1] if markers is not None and len(markers) > 1 else None,
                label=legend_labels[1])
        
    sns.lineplot(x=x, y=y3, 
                color=colors[2] if colors is not None and len(colors) > 2 else None,
                linestyle=linestyles[2] if linestyles is not None and len(linestyles) > 2 else None,
                marker=markers[2] if markers is not None and len(markers) > 2 else None,
                label=legend_labels[2])
    
    # Set labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    
    # Add legend
    plt.legend(loc=legend_loc)
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Display image
    plt.show()
    plt.close()


def plot_performance_across_iterations(G_quan, WGAN=None, show_wgan=False, figsize=(10, 6), 
                                       title='Performance Across Iterations', 
                                       save_path=None):
    """
    Plot model performance across iterations.
    
    Parameters:
        G_quan (numpy.ndarray): Data for WGR model
        WGAN (numpy.ndarray, optional): Data for cWGAN model
        show_wgan (bool): Whether to display the WGAN data
        figsize (tuple): Figure size (width, height)
        title (str): Plot title
        save_path (str, optional): Path to save the figure
    """
    # Create a figure with specified size
    plt.figure(figsize=figsize)
    
    # Plot WGR data
    iterations = np.arange(1, 15000, 100)[:len(G_quan.numpy())]  # Adjust to match data length
    plt.plot(iterations, G_quan.numpy()[:, 2, 0], label='WGR', 
             linestyle='-', linewidth=2, color='#1f77b4')
    
    # Plot WGAN data if provided and requested
    if show_wgan and WGAN is not None:
        wgan_iterations = np.arange(0, 30000, 100)[:len(WGAN.numpy())]  # Adjust to match data length
        plt.plot(wgan_iterations, WGAN.numpy()[:, 2, 0], 
                 label='cWGAN', linestyle='-', linewidth=2, color='#ff7f0e')
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add legend
    plt.legend(frameon=True, fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close()             


# visualization of reconstructed mnist data
def visualize_mnist_digits(dataset, save_path=None, figsize=(6, 25), title=None):
    """
    Visualize MNIST-style dataset, showing two different samples for each digit (0-9).
    
    Parameters:
        dataset: MNIST dataset or similar format (already loaded)
        save_path: If provided, the image will be saved to this path
        figsize: Figure size
        title: Custom title for the figure (default: None)
    """

    # Create a dictionary to store indices by digit
    digit_indices = {i: [] for i in range(10)}
    
    # Iterate through the dataset, collecting indices for each digit
    for idx, (img, label) in enumerate(dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        digit_indices[label].append(idx)
    
    # Create figure with smaller width to make columns closer
    fig, axes = plt.subplots(10, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.00)  # Reduce column spacing
    
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Visualize two samples for each digit
    for digit in range(10):
        # Select two samples for each digit
        selected_indices = digit_indices[digit][:2]
        
        # Ensure there are enough samples
        if len(selected_indices) < 2:
            print(f"Warning: Less than two samples for digit {digit}")
            continue
            
        # Display two samples
        for col in range(2):
            idx = selected_indices[col]
            img, _ = dataset[idx]
            
            # If tensor, convert to numpy array
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.size(0) == 1:  # Single channel image with shape [1, H, W]
                    img = img.squeeze(0).numpy()
                else:
                    img = img.numpy()
            
            # Ensure the image is 2D
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]
                
            # Display image without title
            axes[digit, col].imshow(img, cmap='gray', interpolation='nearest' )
            axes[digit, col].axis('off')  # Turn off axes
    
    plt.tight_layout()  # Adjust layout
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Display image
    plt.show()


def visualize_digits(images, labels, save_path=None, figsize=(6, 16), title=None):
    """
    Visualize custom digit images, showing two samples for each digit (0-9).
    
    Parameters:
        images: Image array with shape [N, H, W] or [N, 1, H, W]
        labels: Label array with shape [N]
        save_path: If provided, the image will be saved to this path
        figsize: Figure size
        title: Custom title for the figure (default: None)
    """
    # Convert images and labels to list format
    data = [(images[i], labels[i]) for i in range(len(images))]
    
    # Create a dataset-like object
    class CustomDataset:
        def __init__(self, data):
            self.data = data
        
        def __getitem__(self, index):
            return self.data[index]
        
        def __len__(self):
            return len(self.data)
    
    # Create custom dataset
    custom_dataset = CustomDataset(data)
    
    # Call visualization function
    visualize_mnist_digits(custom_dataset, save_path, figsize, title)



def visualize_images(images, labels=None, num_cols=5, figsize=None, titles=None, 
                    resize_to=(128, 128), cmap=None, save_path=None):
    """
    Visualize any set of images (color or grayscale) with flexible layout.
    
    Parameters:
        images: List of images or tensor of shape [N, C, H, W] or [N, H, W]
        labels: Optional labels for the images
        num_cols: Number of columns in the grid
        figsize: Figure size (width, height) - calculated automatically if None
        titles: Custom titles for each image (overrides labels)
        resize_to: Target size to resize images (None to keep original size)
        cmap: Colormap for displaying images (None for RGB/BGR images)
        save_path: If provided, the visualization will be saved to this path
    """
    # Convert to list if tensor
    if isinstance(images, torch.Tensor):
        if images.dim() == 4:  # [N, C, H, W]
            images = [img.permute(1, 2, 0).numpy() for img in images]
        elif images.dim() == 3 and images.shape[0] in [1, 3]:  # Single image [C, H, W]
            images = [images.permute(1, 2, 0).numpy()]
        elif images.dim() == 3:  # Batch of grayscale [N, H, W]
            images = [img.numpy() for img in images]
        else:
            raise ValueError(f"Unsupported tensor shape: {images.shape}")
    
    # Handle PIL images
    if isinstance(images[0], Image.Image):
        images = [np.array(img) for img in images]
    
    # Resize images if needed
    if resize_to is not None:
        resizer = transforms.Resize(resize_to, antialias=True)
        resized_images = []
        for img in images:
            # Convert numpy array to tensor for resizing
            if isinstance(img, np.ndarray):
                # Handle different image formats
                if img.ndim == 2:  # Grayscale
                    img_tensor = torch.from_numpy(img).unsqueeze(0).float()
                elif img.ndim == 3:  # RGB/BGR
                    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                else:
                    raise ValueError(f"Unsupported image shape: {img.shape}")
                
                # Resize and convert back to numpy
                resized = resizer(img_tensor)
                if resized.shape[0] == 1:  # Grayscale
                    resized = resized.squeeze(0).numpy()
                else:  # RGB/BGR
                    resized = resized.permute(1, 2, 0).numpy()
                resized_images.append(resized)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        images = resized_images
    
    # Normalize images to [0, 1] if needed
    normalized_images = []
    for img in images:
        if img.dtype == np.uint8:
            normalized_images.append(img / 255.0)
        else:
            # Check if normalization is needed
            if img.max() > 1.0 or img.min() < 0.0:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:  # Avoid division by zero
                    img = (img - img_min) / (img_max - img_min)
            normalized_images.append(img)
    images = normalized_images
    
    # Calculate layout
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division
    
    # Calculate appropriate figure size if not provided
    if figsize is None:
        fig_width = num_cols * 3
        fig_height = num_rows * 3
        figsize = (fig_width, fig_height)
    
    # Create figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Handle the case where there's only one row or column
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.flatten()
    
    # Display images
    for i in range(num_rows * num_cols):
        ax = axes.flat[i] if hasattr(axes, 'flat') else axes[i]
        
        if i < num_images:
            img = images[i]
            
            # Determine colormap
            img_cmap = cmap
            if cmap is None and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)):
                img_cmap = 'gray'
                if img.ndim == 3 and img.shape[2] == 1:
                    img = img.squeeze(2)
            
            # Display image
            ax.imshow(img, cmap=img_cmap)
            
            # Set title if provided
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])
            elif labels is not None and i < len(labels):
                if isinstance(labels[i], torch.Tensor):
                    label = labels[i].item()
                else:
                    label = labels[i]
                ax.set_title(f"Class: {label}")
        
        # Remove axis
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Show figure
    plt.show()  

# example
# random_images = torch.rand(8, 3, 128, 128)  # 8 RGB images
# visualize_images(random_images, num_cols=4, titles=[f"Image {i+1}" for i in range(8)])    

