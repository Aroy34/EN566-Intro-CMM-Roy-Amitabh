import numpy as np
import matplotlib.pyplot as plt

def create_uniform_hpmc_lattice(size, hpmc_loading):
    """Creates a lattice with uniformly distributed HPMC particles."""
    lattice = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            if np.random.rand() < hpmc_loading:
                lattice[x, y] = 1  # HPMC particles
    return lattice

def simulate_gel_layer_expansion(lattice, time_steps, hydration_rate, expansion_rate):
    """Simulates the expansion of HPMC particles into clusters, with clusters forming only near other HPMC particles."""
    size = len(lattice)
    simulation_steps = []

    for step in range(time_steps):
        new_lattice = np.copy(lattice)
        for x in range(size):
            for y in range(size):
                if lattice[x, y] == 1:  # HPMC particle
                    # Check if expansion is possible (adjacent or one step away HPMC particles)
                    expansion_possible = False
                    for dx in [-2, -1, 0, 1, 2]:
                        for dy in [-2, -1, 0, 1, 2]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < size and 0 <= ny < size and lattice[nx, ny] == 1:
                                expansion_possible = True

                    if expansion_possible and np.random.rand() < hydration_rate:
                        # Expand in all four directions
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < size and 0 <= ny < size and np.random.rand() < expansion_rate:
                                    new_lattice[nx, ny] = 2  # Expanding gel layer

        lattice = new_lattice
        simulation_steps.append(new_lattice)

    return simulation_steps



def plot_simulation_steps_with_colorbar(simulation_steps, step_interval=100):
    """Plots the simulation steps at specified intervals with a color bar."""
    for i in range(0, len(simulation_steps), step_interval):
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(simulation_steps[i], interpolation='none')
        ax.set_title(f'Gel Layer Formation - Step {i}')
        ax.axis('off')

        # Adding a color bar to indicate what each color represents
        cbar = plt.colorbar(cax, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['Drug', 'HPMC', 'HPMC Gel'])  # Labels for 0, 1, 2

        # plt.show()

# Parameters for simulation
size = 500  # Lattice size
hpmc_loading = 0.001  # Optimum loading concentration
time_steps = 25  # Total number of time steps
hydration_rate = 0.05  # Hydration rate per time step
expansion_rate = 0.1  # Expansion rate of the gel layer

# Create and run the simulation
uniform_hpmc_lattice = create_uniform_hpmc_lattice(size, hpmc_loading)
simulation_steps_expansion = simulate_gel_layer_expansion(uniform_hpmc_lattice, time_steps, hydration_rate, expansion_rate)

# Plot the simulation steps with color bar
plot_simulation_steps_with_colorbar(simulation_steps_expansion, step_interval=5)
plt.show()


# def create_uniform_hpmc_lattice(size, hpmc_loading):
#     """Creates a lattice with uniformly distributed HPMC particles."""
#     lattice = np.zeros((size, size))
#     for x in range(size):
#         for y in range(size):
#             if np.random.rand() < hpmc_loading:
#                 lattice[x, y] = 1  # HPMC particles
#     return lattice

# def expand_hpmc_particles_once(lattice):
#     """Expand all HPMC particles by one cell in all directions, only once."""
#     size = len(lattice)
#     expanded_lattice = np.zeros_like(lattice)

#     for x in range(size):
#         for y in range(size):
#             if lattice[x, y] == 1:  # HPMC particle
#                 for dx in [-1, 0, 1]:
#                     for dy in [-1, 0, 1]:
#                         nx, ny = x + dx, y + dy
#                         if 0 <= nx < size and 0 <= ny < size:
#                             expanded_lattice[nx, ny] = 1  # Mark expanded area

#     return expanded_lattice

# def label_clusters(lattice):
#     """Label the clusters in the lattice."""
#     size = len(lattice)
#     label = 2  # Starting label for clusters

#     def fill(x, y, current_label):
#         """Recursively fill connected cells with the current label."""
#         if 0 <= x < size and 0 <= y < size and lattice[x, y] == 1:
#             lattice[x, y] = current_label
#             fill(x + 1, y, current_label)
#             fill(x - 1, y, current_label)
#             fill(x, y + 1, current_label)
#             fill(x, y - 1, current_label)

#     for x in range(size):
#         for y in range(size):
#             if lattice[x, y] == 1:
#                 fill(x, y, label)
#                 label += 1

#     return label - 2  # Subtract 2 to exclude '0' (empty) and '1' (unexpanded HPMC)

# # Parameters for the lattice
# size = 50  # Lattice size
# hpmc_loading = 0.6  # Optimum loading concentration

# # Create the HPMC lattice
# uniform_hpmc_lattice = create_uniform_hpmc_lattice(size, hpmc_loading)

# # Expand the HPMC particles once
# expanded_lattice_once = expand_hpmc_particles_once(uniform_hpmc_lattice)

# # Count the number of clusters
# num_clusters = label_clusters(expanded_lattice_once)

# # Output the number of clusters
# num_clusters, expanded_lattice_once

