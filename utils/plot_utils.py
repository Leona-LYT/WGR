import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde_2d(data, cmap='Blues', fill=True, show_cbar=True, thresh=0.05, 
               bw_adjust=0.2, levels=6, fig_size=(8, 6), title=None):
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
    
    Returns:
    fig, ax: The matplotlib figure and axis objects
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
def visualize_mnist_digits(dataset, save_path=None, figsize=(6, 25)):
    """
    Visualize MNIST-style dataset, showing two different samples for each digit (0-9).
    
    Parameters:
        dataset: MNIST dataset or similar format (already loaded)
        save_path: If provided, the image will be saved to this path
        figsize: Figure size
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
            axes[digit, col].imshow(img, cmap='gray')
            axes[digit, col].axis('off')  # Turn off axes
    
    plt.tight_layout()  # Adjust layout
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Display image
    plt.show()

def visualize_custom_digits(images, labels, save_path=None, figsize=(6, 16)):
    """
    Visualize custom digit images, showing two samples for each digit (0-9).
    
    Parameters:
        images: Image array with shape [N, H, W] or [N, 1, H, W]
        labels: Label array with shape [N]
        save_path: If provided, the image will be saved to this path
        figsize: Figure size
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
    visualize_mnist_digits(custom_dataset, save_path, figsize)

#examples
# images_tensor = torch.randn(20, 1, 28, 28)
# sequential_labels = torch.tensor([i//2 for i in range(20)])
# visualize_custom_digits(images_tensor, sequential_labels, save_path="visualization.png")
