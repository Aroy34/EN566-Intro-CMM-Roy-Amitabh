import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse as argp

def gases(X,Y,iterations,trials):
    grid_s =[]
    time_interval = np.linspace(0,iterations,10,dtype=int).tolist()
    
    def lst_update(t1,t2,lst):
        if lst == 0:
            # grid[t1] = 0
            # print(gas_a)
            index_to_replace_a =gas_a.index(t1)
            index_to_replace_empty =empty.index(t2)
            # print(gas_a[index_to_replace])
            gas_a[index_to_replace_a] = t2
            empty[index_to_replace_empty] = t1
            # print(gas_a[index_to_replace])
            # print(t1,t2)
        else:
            # grid[t1] = 0
            # print(gas_b)
            index_to_replace_b =gas_b.index(t1)
            index_to_replace_empty =empty.index(t2)
            # print(gas_a[index_to_replace])
            gas_b[index_to_replace_b] = t2
            empty[index_to_replace_empty] = t1
            # print(gas_b[index_to_replace])
            # print(t1,t2)
    
    def linear_density(grids):
        average_nA_dict = {}
        average_nB_dict = {}

        # Initialize the dictionaries for all keys (time steps) and sections
        for grid in grids:
            for key in grid:
                if key not in average_nA_dict:
                    sections = np.linspace(0, grid[key].shape[0], 10, dtype=int).tolist()
                    average_nA_dict[key] = np.zeros(len(sections) - 1)
                    average_nB_dict[key] = np.zeros(len(sections) - 1)

        # Accumulating the counts for each section and time step over all trials
        for grid in grids:
            for key, grid_array in grid.items():
                sections = np.linspace(0, grid_array.shape[0], 10, dtype=int).tolist()
                for k in range(len(sections) - 1):
                    nA = np.count_nonzero(grid_array[sections[k]:sections[k+1]] == 1)
                    nB = np.count_nonzero(grid_array[sections[k]:sections[k+1]] == -1)
                    average_nA_dict[key][k] += nA
                    average_nB_dict[key][k] += nB

        # Averaging the counts after summing across all trials
        for key in average_nA_dict:
            average_nA_dict[key] /= len(grids)
            average_nB_dict[key] /= len(grids)

        # Plotting the results based on the number of trials
        if len(grids) > 1:
            # Plot only the linear densities for multiple trials
            for key in average_nA_dict:
                plt.figure()
                plt.plot(average_nA_dict[key], label='Average nA')
                plt.plot(average_nB_dict[key], label='Average nB')
                plt.xlabel('Section')
                plt.ylabel('Number of Particles')
                plt.title(f"Average Linear Density per Section at Time Step {key}")
                plt.legend()
                plt.savefig(f"Average_Linear_Density_Grid_After_{key}_Steps_{1}_Trial.png")
                # plt.show()
        else:
            # Plot both the linear densities and grid state for a single trial
            for key in average_nA_dict:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Plot for average number of particles
                axes[0].plot(average_nA_dict[key], label='Average nA')
                axes[0].plot(average_nB_dict[key], label='Average nB')
                axes[0].set_xlabel('Section')
                axes[0].set_ylabel('Number of Particles')
                axes[0].set_title(f"Linear Density of Particles per Section at Time Step {key}")
                axes[0].legend()

                # Plot for grid state
                cmap = ListedColormap(['red', 'white', 'blue'])
                average_grid = np.mean([grid[key] for grid in grids], axis=0)
                axes[1].imshow(np.transpose(average_grid), cmap=cmap)
                axes[1].set_title(f"Grid State at Time Step {key}")

                plt.tight_layout()
                plt.savefig(f"Linear_Density_Grid_After_{key}_Steps_{1}_Trial.png")
                # plt.show()
                
    def pos_check (x ,y):
        if x < X and x>= 0 and y < Y and y>= 0:
            return True
        
    for i in range(1,trials+1):
        print(i)
        # Grid making
        grid = np.zeros((X,Y))
        print(grid)
        gas_a = []
        gas_b =[]
        empty = []
 
        for i in range(0,int(x/3)):
            for j in range(0,y):
                grid[i][j] = 1
                gas_a.append((i,j))
        for i in range(int(2*x/3),x):
            for j in range(0,y):
                grid[i][j] = -1
                gas_b.append((i,j))
        for i in range(int(x/3),int(2*x/3)):
            for j in range(0,y):
                grid[i][j] = 0
                empty.append((i,j))
        # print(gas_a)
        
        n = 0
                
        grid_dic = {}
        
        while n < iterations:
            
            a_or_b = random.randint(0,1)
            # print(a_or_b)
            if a_or_b == 0:
                i , j  = random.choice(gas_a)
                # print(i,j)
            if a_or_b == 1:
                i , j  = random.choice(gas_b)
                # print(i,j)
 
            # stop = True
            # for j in range(10):
            direction = [(1,0), (0,1), (-1,0), (0,-1)]
            x_, y_ = random.choice(direction)
            # if (i+x_,j+y_) in empty:
            #     print(i+x_,j+y_,"isnide")
            
            if pos_check(i+x_,j+y_):
                if grid [i+x_][j+y_] == 0:
                    # print(grid[i+x_][j+y_], "changed to",grid[i][j])
                    grid[i+x_][j+y_] = grid[i][j]
                    grid[i][j] = 0
                    # print( i+x_,j+y_,i,j)
                    lst_update((i,j),(i+x_,j+y_),a_or_b)
                    # print(i,j)
                    # break
            if n in time_interval:
                grid_dic[n] = np.copy(grid)
                # print(n)
                # print(grid)
            n = n+1   
        
        grid_s.append(grid_dic)
    
    # print(grid_s)
    
    linear_density(grid_s)
            
if __name__ == "__main__":
    x = 60
    y = 40
    iterations = 10
    psr = argp.ArgumentParser("gases")
    psr.add_argument('--part', type=str, default="1,2,3",
                     help="enter the part , ")  
    arg = psr.parse_args()
    part_str = arg.part.split(",")
    part_list = [int(pt) for pt in part_str]  # list fo all the step widths 
    
    for part in part_list:
        if part == 1 or part == 2:
            trials = 1
            gases(x,y,iterations,trials)
        elif part ==3:
            trials = 100
            gases(x,y,iterations,trials)
