import numpy as np
import matplotlib.pyplot as plt

lattice_size = 100
probabilities = np.linspace(0.1, 0.9, 

# Function to initialize the lattice
def initialize_lattice(size):
    return np.zeros((size, size))

# Function to randomly occupy 
def occupy_site(probability):
    return np.random.rand() < probability

# Function to find the largest cluster
def find_largest_cluster(lattice):
    unique, counts = np.unique(lattice, return_counts=True)
    
    cluster_sizes = {}
    for i in range(len(unique)):
        if unique[i] != 0:  # Skip the background label 0
            cluster_sizes[unique[i]] = counts[i]

    largest_cluster_label = None
    max_size = 0
    for label, size in cluster_sizes.items():
        if size > max_size:
            max_size = size
            largest_cluster_label = label

    return largest_cluster_label

# Function to label the clusters 
def label_clusters(lattice, prob):
    current_label = 1
    for i in range(lattice_size):
        for j in range(lattice_size):
            if lattice[i, j] == -1:  
                neighbors = []
                if i > 0:
                    neighbors.append(lattice[i-1, j])
                if i < lattice_size-1:
                    neighbors.append(lattice[i+1, j])
                if j > 0:
                    neighbors.append(lattice[i, j-1])
                if j < lattice_size-1:
                    neighbors.append(lattice[i, j+1])
                
                neighbors = [n for n in neighbors if n > 0]  # remove unoccupied
                if neighbors:
                    min_label = min(neighbors)
                    lattice[i, j] = min_label
                    # Merge clusters
                    for n in neighbors:
                        if n != min_label:
                            for x in range(lattice_size):
                                for y in range(lattice_size):
                                    if lattice[x, y] == n:
                                        lattice[x, y] = min_label
                else:
                    lattice[i, j] = current_label
                    current_label += 1
    return lattice

# Function to visualize the lattice
def visualize_lattice(lattices, probabilities):
    for i in range(len(lattices)):
        lattice = lattices[i]
        largest_cluster_label = find_largest_cluster(lattice)
        if largest_cluster_label:
            largest_cluster = np.zeros(lattice.shape, dtype=bool)
            for x in range(lattice_size):
                for y in range(lattice_size):
                    if lattice[x, y] == largest_cluster_label:
                        largest_cluster[x, y] = True
            overlay = np.zeros((lattice_size, lattice_size, 3), dtype=np.uint8)
            for x in range(lattice_size):
                for y in range(lattice_size):
                    if lattice[x, y] != 0:
                        overlay[x, y] = [255, 255, 255] 
                    if largest_cluster[x, y]:
                        overlay[x, y] = [255, 0, 0]  
            plt.title(f'p = {probabilities[i]:.2f}')
            plt.imshow(overlay, interpolation='none')
            plt.savefig(f'p = {probabilities[i]:.2f}.png')

if __name__ == "__main__":
    lattices = []
    for p in probabilities:
        lattice = initialize_lattice(lattice_size)
        for i in range(lattice_size):
            for j in range(lattice_size):
                if occupy_site(p):
                    lattice[i, j] = -1 
        labeled_lattice = label_clusters(lattice, p)
        lattices.append(labeled_lattice)

    visualize_lattice(lattices, probabilities)
