import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
import argparse as argp


def initial_config(N):
    return np.random.choice([-1, 1], size=(n,n))

def calculate_energy(config, J):
    energy = 0
    n = len(config)
    for i in range(n):
        for j in range(n):
            S = config[i, j]
            nb = config[(i+1)%n, j] + config[i, (j+1)%n] + config[(i-1)%n, j] + config[i, (j-1)%n]
            energy += -J * nb * S
    return energy / 4

def calculate_delta_E(config, i, j, J):
    n = len(config)
    s = config[i, j]
    nb = config[(i+1)%n, j] + config[i, (j+1)%n] + config[(i-1)%n, j] + config[i, (j-1)%n]
    return 2 * J * s * nb

def calculate_magnetization(config):
    return np.sum(config)


def metropolis_step(config, T, J):
    N = len(config)
    delta_E_values = []  # To store ΔE values for each step
    for i in range(N):
        for j in range(N):
            delta_E = calculate_delta_E(config, i, j, J)
            delta_E_values.append(delta_E)
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (kB * T)):
                config[i, j] *= -1
    return config, delta_E_values

def simulate(L, T, J, mc_steps, kB):
    ini_config = initial_config(L)
    energies = []
    magnetizations = []
    specific_heats = []
    final_configs = {}

    for t in T:
        config = np.copy(ini_config)
        print(t)
        E = M = E2 = 0  # E2 for energy squared
        for _ in range(mc_steps):
            config, _ = metropolis_step(config, t, J)
            energy = calculate_energy(config, J)
            E += energy
            E2 += energy**2
            M += calculate_magnetization(config)

        E_avg = E / mc_steps
        E2_avg = E2 / mc_steps
        M_avg = M / mc_steps

        energies.append(E_avg)
        magnetizations.append(M_avg)
        final_configs[t] = np.copy(config)

        delta_E = np.sqrt(E2_avg - E_avg**2)
        specific_heat = delta_E**2 / (kB * t**2)
        specific_heats.append(specific_heat)

    return energies, magnetizations, specific_heats, final_configs



def plotter_first(T, energies, magnetizations, specific_heats, final_configs, n):


    # Energy vs Temperature Plot
    plt.figure()
    plt.plot(T, energies, 'o-')
    plt.title('Energy vs Temperature')
    plt.xlabel('Temperature (1/kB)')
    plt.ylabel('Energy')
    plt.legend()
    plt.savefig("Energy_vs_Temperature.png")

    # Magnetization vs Temperature Plot
    plt.figure()
    plt.plot(T, magnetizations, 'o-')
    plt.title('Magnetization vs Temperature')
    plt.xlabel('Temperature (1/kB)')
    plt.ylabel('Magnetization')
    plt.legend()
    plt.savefig("Magnetization_vs_Temperature.png")

    # Specific Heat vs Temperature Plot
    plt.figure()
    plt.plot(T, specific_heats, 'o-')
    plt.title('Specific Heat vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.legend()
    plt.savefig("Specific_Heat_vs_Temperature.png")

    # Plotting heatmaps of the final configurations for selected temperatures
    for temp, config in final_configs.items():
        cmap = ListedColormap(['saddlebrown', 'navy'])
        plt.figure(figsize=(6, 6))
        plt.imshow(config, cmap=cmap, vmin=-1, vmax=1)
        plt.title(f'Spin Configuration at T={temp:.2f}')
        plt.colorbar(label='Spin')
        plt.savefig(f"Spin_Configuration_at_T={temp:.2f}.png")

    plt.show()
    

if __name__ == "__main__":
    mc_steps = 10**3 # Monte Carlo steps per temperature
    kB = 1   # Boltzmann constant
    J = 1.5 # Interaction strength
    
    psr = argp.ArgumentParser("isinng")
    psr.add_argument('--part', type=str, default="1,2",
                     help="enter the part , ")  
    arg = psr.parse_args()
    part_str = arg.part.split(",")
    part_list = [int(pt) for pt in part_str]  # list fo all the step widths 
    for part in part_list:
        if part == 1:
            n = 10  # Lattice size
            T = np.linspace(0.1, 5, 10) # Temperature range
            energies, magnetizations, specific_heats, final_configs = simulate(n, T, J, mc_steps, kB)
            energies = np.array(energies)
            specific_heats = np.array(specific_heats)
            plotter_first(T,energies,magnetizations, specific_heats, final_configs,n)
      
        elif part == 2:
            print("Need to add code")
            pass
            
    
    