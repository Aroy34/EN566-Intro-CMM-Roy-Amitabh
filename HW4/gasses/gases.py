import random
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap


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
        # Assuming 'grids' is a list of dictionaries with keys being time steps and values being numpy arrays representing grids.
        average_dict = {}
        # print("helooo",len(grids))

        # Sum values for each key (time step)
        for grid in grids:
            for key, grid_array in grid.items():
                if key not in average_dict:
                    average_dict[key] = grid_array
                else:
                    average_dict[key] += grid_array

        # Average the values
        for key in average_dict:
            average_dict[key] = average_dict[key] / len(grids)
        
        # print("hello",average_dict)

        # Plotting each key (time step)
        for n, grid_array in average_dict.items():
            num_rows, num_columns = grid_array.shape
            sections = np.linspace(0, num_rows, 10, dtype=int).tolist()
            nA = []
            nB = []
            print("enterdn")

            for k in range(len(sections) - 1):
                A = 0
                B = 0
                for i in range(sections[k], sections[k+1]):
                    A += np.count_nonzero(grid_array[i] == 1)
                    B += np.count_nonzero(grid_array[i] == -1)

                total = A + B if A + B > 0 else 1
                nA.append(A / total)
                nB.append(B / total)
            
            # print(nA)
            if len(grids) == 1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.plot(sections[:-1], nA, label='nA')
                ax1.plot(sections[:-1], nB, label='nB')
                ax1.set_xlabel('Sections')
                ax1.set_ylabel('nA / nB')
                ax1.legend()
                
                cmap = ListedColormap(['red', 'white', 'blue'])
                ax2.imshow(np.transpose(grid_array), cmap=cmap)
                ax2.set_title(f"Time-Interval = {n}")

            if len(grids) > 1: 
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
                # print("ran")
                ax1.plot(sections[:-1], nA, label='nA')
                ax1.plot(sections[:-1], nB, label='nB')
                ax1.set_xlabel('Sections')
                ax1.set_ylabel('nA / nB')
                ax1.legend()
            
                    
                
            
            plt.savefig(f"Linear-Density_Grid_after_{n}_steps_{len(grids)}_trials.png")
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
        
        # print(time_interval)
        
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
                print("Pushing to the dic")
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
    iterations = 10**5
    trials = [1]
    for i in  range(len(trials)):
        gases(x,y,iterations,trials[i])