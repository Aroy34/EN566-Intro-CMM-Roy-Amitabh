import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements

# Define the function to create the lattice with HPMC particles
def create_lattice(size, loading):
    lattice = np.zeros((size, size))
    num_particles = int(loading * size * size)
    for _ in range(num_particles):
        x, y = np.random.randint(0, size, 2)
        lattice[x, y] = 1
    return lattice

# Define the function to expand the particles in the lattice
def expand_particles(lattice):
    size = lattice.shape[0]
    expanded = np.copy(lattice)
    for x in range(size):
        for y in range(size):
            if lattice[x, y] == 1:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if 0 <= x+dx < size and 0 <= y+dy < size:
                            expanded[x+dx, y+dy] = 1
    return expanded

# Define the function to calculate the average cluster size
def calculate_average_cluster_size(lattice):
    lw, num = measurements.label(lattice)
    area = measurements.sum(lattice, lw, index=np.arange(lw.max() + 1))
    area = area[area > 0]
    return np.mean(area)

# Simulation parameters
size = 50  # Size of the lattice (e.g., 50x50 for quicker computation)
loadings = np.linspace(0.05, 1, 10)  # Range of loading percentages
average_cluster_sizes = []

# Set up the plot for the lattice layouts
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))  # Adjust nrows and ncols based on the number of loadings
axes = axes.flatten()  # Flatten the array for easy indexing

# Run the simulation for different loadings and calculate average cluster sizes
for i, loading in enumerate(loadings):
    lattice = create_lattice(size, loading)
    expanded_lattice = expand_particles(lattice)
    avg_cluster_size = calculate_average_cluster_size(expanded_lattice)
    average_cluster_sizes.append(avg_cluster_size)

    # Plot the lattice layout
    axes[i].imshow(expanded_lattice, cmap='viridis', interpolation='none')
    axes[i].set_title(f'Loading: {loading:.2f}')
    axes[i].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the lattice layout plots
plt.show()

# Plot the results for average cluster size
plt.figure(figsize=(10, 6))
plt.plot(loadings * 100, average_cluster_sizes, 'o-', label='Average Cluster Size')
plt.xlabel('% Loading of HPMC Particles')
plt.ylabel('Average Cluster Size')
plt.title('Average Cluster Size vs % Loading of HPMC Particles')
plt.legend()
plt.grid(True)
plt.show()
