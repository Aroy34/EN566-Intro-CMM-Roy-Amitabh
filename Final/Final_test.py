import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def create_lattice(size, concentration):
    """ Create a lattice with randomly distributed HPMC particles. """
    lattice = np.random.rand(size, size) < concentration
    return lattice

def check_percolation(lattice):
    """ Check if a percolating path exists in the lattice. """
    # Implement a simple percolation checking algorithm
    # This can be a depth-first search or union-find algorithm
    pass

def higuchis_model(concentration, a, b):
    """ Higuchi's model for drug release. """
    return a * np.sqrt(concentration) + b

def simulate_drug_release(size, concentrations):
    """ Simulate drug release for different HPMC concentrations. """
    release_rates = []
    for conc in concentrations:
        lattice = create_lattice(size, conc)
        if check_percolation(lattice):
            # Fit Higuchi's model to get release rate
            popt, _ = curve_fit(higuchis_model, conc, some_empirical_data)
            release_rate = higuchis_model(conc, *popt)
            release_rates.append(release_rate)
        else:
            release_rates.append(0)  # No release if no percolation
    return release_rates



# Parameters
lattice_size = 100
hpmc_concentrations = np.linspace(0.1, 0.6, 50)  # Example range

# Simulation
release_rates = simulate_drug_release(lattice_size, hpmc_concentrations)

# Plotting the results
plt.plot(hpmc_concentrations, release_rates)
plt.xlabel('HPMC Concentration')
plt.ylabel('Drug Release Rate')
plt.title('Drug Release Rate vs HPMC Concentration')
plt.show()


class Lattice:
    def __init__(self, size, concentration):
        """
        Initialize the lattice.

        :param size: int, the size of the lattice (size x size).
        :param concentration: float, the concentration of HPMC particles.
        """
        self.size = size
        self.concentration = concentration
        self.lattice = self._create_lattice()

    def _create_lattice(self):
        """
        Create a lattice with randomly distributed HPMC particles.

        :return: np.array, a 2D array representing the lattice.
        """
        return np.random.rand(self.size, self.size) < self.concentration

    def display(self):
        """
        Display the lattice structure.
        """
        import matplotlib.pyplot as plt
        plt.imshow(self.lattice, cmap='Greys')
        plt.title(f'Lattice with HPMC concentration: {self.concentration}')
        plt.show()

class Percolation:
    def __init__(self, lattice):
        """
        Initialize the Percolation class with a given lattice.

        :param lattice: Lattice, an instance of the Lattice class.
        """
        self.lattice = lattice.lattice
        self.size = lattice.size
        self.percolates = self._check_percolation()

    def _check_percolation(self):
        """
        Check if a percolating path exists in the lattice.

        :return: bool, True if percolation occurs, False otherwise.
        """
        visited = set()

        def dfs(x, y):
            """ Depth-first search to check percolation. """
            if (x, y) in visited or not (0 <= x < self.size and 0 <= y < self.size):
                return False
            visited.add((x, y))
            if self.lattice[x, y] == 0:
                return False
            if x == self.size - 1:
                return True
            return (dfs(x + 1, y) or dfs(x - 1, y) or
                    dfs(x, y + 1) or dfs(x, y - 1))

        for y in range(self.size):
            if dfs(0, y):
                return True
        return False

    def is_percolating(self):
        """
        Return the percolation status.

        :return: bool, the percolation status.
        """
        return self.percolates

class DrugReleaseSimulation:
    def __init__(self, lattice_size, concentration_range):
        """
        Initialize the drug release simulation.

        :param lattice_size: int, the size of the lattice.
        :param concentration_range: list, range of HPMC concentrations to simulate.
        """
        self.lattice_size = lattice_size
        self.concentration_range = concentration_range
        self.release_rates = []

    def higuchis_model(self, concentration, a, b):
        """
        Higuchi's model for drug release.

        :param concentration: float, HPMC concentration.
        :param a, b: float, model parameters.
        :return: float, estimated drug release rate.
        """
        return a * np.sqrt(concentration) + b

    def run_simulation(self):
        """
        Run the drug release simulation over the specified concentration range.
        """
        for conc in self.concentration_range:
            lattice = Lattice(self.lattice_size, conc)
            percolation = Percolation(lattice)

            if percolation.is_percolating():
                # Assuming some empirical data for fitting
                # In real scenarios, this data should come from experimental observations
                empirical_data = [self.higuchis_model(c, 1, 0) for c in self.concentration_range]
                popt, _ = curve_fit(self.higuchis_model, self.concentration_range, empirical_data)
                release_rate = self.higuchis_model(conc, *popt)
                self.release_rates.append(release_rate)
            else:
                self.release_rates.append(0)  # No release if no percolation

    def plot_results(self):
        """
        Plot the simulation results.
        """
        plt.plot(self.concentration_range, self.release_rates, marker='o')
        plt.xlabel('HPMC Concentration')
        plt.ylabel('Drug Release Rate')
        plt.title('Drug Release Rate vs HPMC Concentration')
        plt.grid(True)
        plt.show()

class SimulationController:
    def __init__(self, lattice_size, concentration_range):
        """
        Initialize the SimulationController.

        :param lattice_size: int, the size of the lattice.
        :param concentration_range: list, range of HPMC concentrations to simulate.
        """
        self.lattice_size = lattice_size
        self.concentration_range = concentration_range
        self.simulation = DrugReleaseSimulation(lattice_size, concentration_range)

    def run(self):
        """
        Run the entire simulation process.
        """
        self.simulation.run_simulation()
        self.simulation.plot_results()

    def display_lattice_at_concentration(self, concentration):
        """
        Display the lattice structure at a specific HPMC concentration.

        :param concentration: float, the HPMC concentration.
        """
        lattice = Lattice(self.lattice_size, concentration)
        lattice.display()

if __name__ == "__main__":
    # Define the parameters for the simulation
    lattice_size = 100  # Size of the lattice
    concentration_range = np.linspace(0.1, 1.1, 50)  # Range of HPMC concentrations to simulate

    # Create an instance of the SimulationController with the defined parameters
    controller = SimulationController(lattice_size, concentration_range)

    # Run the entire simulation
    controller.run()

    # Optionally, display the lattice structure at a specific concentration
    # You can change the concentration value to see different results
    controller.display_lattice_at_concentration(0.1)
    controller.display_lattice_at_concentration(0.3)
    controller.display_lattice_at_concentration(0.6)
    controller.display_lattice_at_concentration(0.9)
