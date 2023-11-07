import numpy as np
import matplotlib.pyplot as plt

# Function to plot the distribution with different subdivisions
def plot_distribution(data, subdivisions, title):
    for sub in subdivisions:
        plt.figure()  # This creates a new figure for each plot
        plt.hist(data, bins=sub, density=True, alpha=0.5, label=f'{sub} bins')
        plt.title(title + f" - {sub} subdivisions")
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.show()

# Part 1: Uniform distribution
def uniform_distribution(sample_size):
    # Generate random numbers
    data = np.random.rand(sample_size)
    
    # Plot with different subdivisions
    plot_distribution(data, [10, 20, 50, 100], f'Uniform Distribution with {sample_size} samples')

# Part 2: Gaussian distribution
def gaussian_distribution(sample_size, sigma):
    # Generate random numbers using Box-Muller transform
    u1, u2 = np.random.rand(2, sample_size//2)  # Corrected to ensure we get the right number of samples
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2) * sigma
    
    # Plot with different subdivisions
    plot_distribution(z0, [10, 20, 50, 100], f'Gaussian Distribution with {sample_size} samples')

    # Overlay the theoretical Gaussian distribution
    plt.figure()
    count, bins, ignored = plt.hist(z0, bins=100, density=True, alpha=0.5, label='Generated distribution')
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins**2)/(2 * sigma**2)), linewidth=2, color='r', label='Theoretical distribution')
    plt.title(f'Gaussian Distribution with {sample_size} samples (Overlay)')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.show()

# Execute for 1,000 and 1,000,000 samples for uniform distribution
for size in [1000, 1000000]:
    uniform_distribution(size)

# Execute for 1,000 and 1,000,000 samples for Gaussian distribution
for size in [1000, 1000000]:
    gaussian_distribution(size, sigma=1.0)
