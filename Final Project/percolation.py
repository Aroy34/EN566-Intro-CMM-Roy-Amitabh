import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements

# Define the function to create the lattice with HPMC particles
def create_lattice(size, loading):
    lattice = np.zeros((size, size)) # Empty lattice with drugs
    num_particles = int(loading * size * size)
    for particle in range(num_particles):
        run = True
        while run:
            x, y = np.random.randint(0, size, 2)
            if lattice[x, y] == 0:
                run = False
        lattice[x, y] = 1 # Addition of HPMC
    return lattice

def function_that_calculates_the_swellign_of_HPMC():
    pass
def fucntion_that_calculates_number_of_cluster():
    pass

if __name__ == "__main__":
    size = 50  
    loadings = np.linspace(0.05, 1, 10)  #loading percentages

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))  
    axes = axes.flatten()  

    for i, loading in enumerate(loadings):
        lattice = create_lattice(size, loading)

        # Plot the lattice layout
        axes[i].imshow(lattice)
        axes[i].set_title(f'Loading: {loading:.2f}')
        axes[i].axis('off')

    plt.tight_layout()

    plt.show()
    
    ###### Finally plot number of clusters vs % loading ######


