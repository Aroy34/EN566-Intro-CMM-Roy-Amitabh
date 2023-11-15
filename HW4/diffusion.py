import numpy as np
import matplotlib.pyplot as plt
import argparse as argp

def diffusion(D, boundary,initial_spread, dx, dt,num_itr ):
    # Create spatial grid
    
    x = np.arange(-boundary/2, boundary/2, dx)
    print(x)
    grid_size = len(x)
    
    # Initial Density Profile
    
    rho = []
    for x_elements in x:
        if abs(x_elements) < initial_spread:
            rho.append(1)
        else:
            rho.append(0)
            
            
    def normal_distribution(x, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma)**2)
    
    plot_steps = np.linspace(100,num_itr,5)
    
    plt.figure(figsize=(10, 6))
    # Time evolution of the density profile
    for n in range(0, num_itr):
        rho_new = np.zeros(grid_size).tolist()
        
        for i in range(grid_size):
            if i - 1 >= 0:
                rho_left = rho[i - 1] 
            else:
                rho_left = 0 
            if i + 1 < grid_size:
                rho_right = rho[i + 1] 
            else:
                rho_right = 0
            rho_new[i] = rho[i] + D * dt / dx**2 * (rho_left - 2 * rho[i] + rho_right)
        rho = rho_new

        
        if n in plot_steps:
            sigma_t = np.sqrt(2 * D * n * dt)
            plt.plot(x, rho / (np.array(rho).sum() * dx), label=f'Numerical t = {n*dt} s')  #
            plt.plot(x, normal_distribution(x, sigma_t), linestyle='--', label=f'Sigma= {sigma_t:.2f} at time =  {n*dt} s ')
            
        
    plt.xlabel(' Distance from the source')
    plt.ylabel('Density')
    plt.title('1D Diffusion Equation Solution')
    plt.legend()
    plt.savefig("1D Diffusion Equation Solution")
    plt.show()

if __name__ == "__main__":
    # Parameters
    D = 2
    boundary = 10
    initial_spread = 0.25
    dx = 0.1
    dt = 0.0001
    num_itr = 1000

    # Run the solver
    psr = argp.ArgumentParser("diffusion")
    psr.add_argument('--part', type=str, default="1,2",
                     help="enter the part , ")  
    arg = psr.parse_args()
    part_str = arg.part.split(",")
    part_list = [int(pt) for pt in part_str]  # list fo all the step widths 
    
    for part in part_list:
            if part == 1 or part == 2:
                diffusion(D, boundary, initial_spread,dx, dt, num_itr)
