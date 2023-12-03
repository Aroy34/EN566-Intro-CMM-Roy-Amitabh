import numpy as np
import matplotlib.pyplot as plt
import argparse as argp
import numba
from numba import njit
from scipy.signal import find_peaks


#Using numba to do JIT compilation
@njit
def initial_config(n):
    """Giving lattic either the up and down spin with 50% probability """
    config = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if np.random.random() < 0.5:
                config[i, j] = 1 
            else:
                config[i, j] = -1
    return config

#Using numba to do JIT compilation
@numba.jit(nopython=True)
def calculate_energy_vectorized(config, J):
    """Calculating the energy fo the system by sweeping through each coordinates"""
    n = config.shape[0]
    energy = 0
    for i in range(n):
        for j in range(n):
            # centre coordinate
            s = config[i, j]
            # Surrounding coordinates[Periodic Boundary Condition]
            nb = config[(i+1)%n, j] + config[i, (j+1)%n] + \
                 config[(i-1)%n, j] + config[i, (j-1)%n]
            energy += -J * s * nb
    return energy


#Using numba to do JIT compilation
@numba.jit(nopython=True)
def calculate_magnetization(config):
    """Calculating the magnetisation of the system adn then returning the absolute value"""
    return np.abs(np.sum(config))

#Using numba to do JIT compilation
@numba.jit(nopython=True)
def metropolis_step_vectorized(config, T, J, kB, n):
    for itr in range(n**2):
        # Randomly selecting the coordinates
        i = np.random.randint(n)
        j = np.random.randint(n)
        s = config[i, j]
        nb = config[(i+1)%n, j] + config[i, (j+1)%n] + \
             config[(i-1)%n, j] + config[i, (j-1)%n]
        delta_E = 2 * J * s * nb
        # Reverting the flip if conditions meet the criteria
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (kB * T)):
            config[i, j] *= -1
    return config

# @numba.jit(nopython=True)
def simulate(n, T, J, mc_steps, kB):
    ini_config = initial_config(n)
    energies = []
    magnetizations = []
    specific_heats = []
    final_configs = {}

    for t in T:
        print("T -> " ,t)
        # Making copy of the array toa void changing the original copy
        config = np.copy(ini_config)
        temp_energies = []  # list to store energies for this temperature

        for sweep_no in range(mc_steps):
            config = metropolis_step_vectorized(config, t, J, kB, n)
            energy = calculate_energy_vectorized(config, J)
            temp_energies.append(energy)

        # Calculate the average energy for this temperature
        E_avg = np.mean(np.array(temp_energies)) 
        E2_avg = np.mean(np.array(temp_energies)**2) 
        delta_E_2 = (E2_avg - E_avg**2) 
        specific_heat = delta_E_2 / (kB * t**2)

        # Append the average results for this temperature
        energies.append(E_avg)
        M = calculate_magnetization(config) / n**2
        magnetizations.append(M)
        specific_heats.append(specific_heat)
        final_configs[t] = np.copy(config)

    return energies, magnetizations, specific_heats, final_configs

def find_peak_derivative(T, y_values):
    # Calculate the first derivative of y with respect to T
    dydT = np.gradient(y_values, T)
    # Find the index of the maximum value in the first derivative
    max_dydT_index = np.argmax(np.abs(dydT))
    # Return the temperature and value of the derivative at this point
    return T[max_dydT_index], dydT[max_dydT_index]

def plotter_first(T, energies, magnetizations, specific_heats, final_configs, n):
    # Find the temperatures corresponding to the maximum derivative for energy and magnetization
    peaks, properties = find_peaks(specific_heats)
    Tc_from_specific_heat = T[peaks][-1]  # Assuming the last peak corresponds to Tc, if are some noise
    Tc_from_energy, max_dEdT = find_peak_derivative(T, energies)
    Tc_from_magnetization, max_dMdT = find_peak_derivative(T, magnetizations)

    # Energy vs Temperature Plot
    plt.figure()
    plt.plot(T, energies, 'o-')
    plt.axvline(x=Tc_from_energy, color='r', linestyle='--', label=f'Tc={Tc_from_energy:.2f}')
    plt.title('Energy vs Temperature')
    plt.xlabel('Temperature (1/kB)')
    plt.ylabel('Energy')
    plt.legend()
    plt.savefig("Energy_vs_Temperature.png")
    plt.close()

    # Magnetization vs Temperature Plot
    plt.figure()
    plt.plot(T, magnetizations, 'o-')
    plt.axvline(x=Tc_from_magnetization, color='r', linestyle='--', label=f'Tc={Tc_from_magnetization:.2f}')
    plt.title('Magnetization vs Temperature')
    plt.xlabel('Temperature (1/kB)')
    plt.ylabel('Magnetization')
    plt.legend()
    plt.savefig("Magnetization_vs_Temperature.png")
    # plt.show()
    plt.close()

    # Specific Heat vs Temperature Plot
    plt.figure()
    plt.plot(T, specific_heats, 'o-')
    plt.axvline(x=Tc_from_specific_heat, color='r', linestyle='--', label=f'Tc={Tc_from_specific_heat:.2f}')
    plt.title('Specific Heat vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.legend()
    plt.savefig("Specific_Heat_vs_Temperature.png")
    plt.close()

    # Plotting heatmaps of the final configurations for selected temperatures
    for temp, config in final_configs.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(config, cmap='jet', vmin=-1, vmax=1)
        plt.title(f'Spin Configuration at T={temp:.2f}')
        plt.colorbar(label='Spin')
        plt.savefig(f"Spin_Configuration_at_T={temp:.2f}.png")
        plt.close()

    # plt.show()

if __name__ == "__main__":
    mc_steps = 10**5 # Monte Carlo steps per temperature
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
            n = 50  # Lattice size
            T = np.linspace(1, 5, 30) # Temperature range
            energies, magnetizations, specific_heats, final_configs = simulate(n, T, J, mc_steps, kB)
            energies = np.array(energies)
            specific_heats = np.array(specific_heats)
            plotter_first(T,energies,magnetizations, specific_heats, final_configs,n)
      
        elif part == 2:
            pass
        