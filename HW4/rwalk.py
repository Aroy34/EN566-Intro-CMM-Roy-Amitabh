import random
import numpy as np
import matplotlib.pyplot as plt
import argparse as argp


def rwalk():
    
    def walk(steps, origin_x, origin_y):
        """ Perform a 2D random walk """
        x, y = origin_x, origin_y
        x_positions, y_positions = [x], [y]
        
        for _ in range(steps):
            rand = random.uniform(0, 1)
            if rand <= 0.25:
                dx, dy = 1, 0
            elif rand <= 0.5:
                dx, dy = -1, 0
            elif rand <= 0.75:
                dx, dy = 0, 1
            else:
                dx, dy = 0, -1

            x += dx 
            y += dy
            x_positions.append(x)
            y_positions.append(y)

        return x_positions[1:], y_positions[1:]  # removing the 

    def simulate_walks(steps, num_walks, start_x, start_y):
        all_x_positions = np.zeros(steps)
        all_y_positions = np.zeros(steps)
        all_x_squared = np.zeros(steps)
        all_y_squared = np.zeros(steps)

        for i in range(num_walks):
            x_pos, y_pos = walk(steps, start_x, start_y)
            all_x_positions = np.array(x_pos)
            all_y_positions += np.array(y_pos)
            all_x_squared += np.array(x_pos)**2
            all_y_squared += np.array(y_pos)**2

        return (all_x_positions / num_walks, 
                all_y_positions / num_walks, 
                all_x_squared / num_walks, 
                all_y_squared / num_walks)
        
    psr = argp.ArgumentParser("rwalk")
    psr.add_argument('--part', type=str, default="1,2",
                     help="enter the part , ")  
    arg = psr.parse_args()
    part_str = arg.part.split(",")
    part_list = [int(pt) for pt in part_str]  # list fo all the step widths 
    
    # First 3 steps with a single walk
    x_pos_3, y_pos_3 = walk(3, 0, 0)

    # Remaining 97 steps with 10000 walks
    num_walks = 10000
    steps = 97  # 97 additional steps
    x_avg_97, y_avg_97, x_sq_avg_97, y_sq_avg_97 = simulate_walks(steps, num_walks, x_pos_3[-1], y_pos_3[-1])

    # Combining the results
    total_steps = 100  # including the initial position
    x_avg = np.concatenate(([0] + x_pos_3, x_avg_97))
    y_avg = np.concatenate(([0] + y_pos_3, y_avg_97))
    x_sq_avg = np.concatenate(([0**2] + [x**2 for x in x_pos_3], x_sq_avg_97))
    y_sq_avg = np.concatenate(([0**2] + [y**2 for y in y_pos_3], y_sq_avg_97))

    # Mean square distance from origin
    mean_square_distance = x_sq_avg + y_sq_avg

    for part in part_list:
        if part == 1:
            plt.figure(1)
            plt.plot(range(total_steps + 1), x_avg, label='<Xn>')
            plt.title('X Position (<Xn>)')
            plt.xlabel('Steps')
            plt.ylabel('Xn')
            plt.savefig("(<Xn>).pdf")
            plt.legend()

            # Plotting <Xn^2>
            plt.figure(2)
            plt.plot(range(total_steps + 1), x_sq_avg, label='<Xn^2>')
            plt.title('Mean Square X Position (<Xn^2>)')
            plt.xlabel('Steps')
            plt.ylabel('Xn^2')
            plt.savefig("(<Xn>)^2.pdf")
            plt.legend()
            # Plotting <r^2>

        elif part == 2:
            plt.figure(3)
            n = np.arange(total_steps + 1)
            slope, intercept = np.polyfit(n, mean_square_distance, 1)
            plt.plot(range(total_steps + 1), mean_square_distance,label = f"Slope = {round(slope,4)}")
            plt.title('Mean Square Distance (<r^2>)')
            plt.xlabel('Steps')
            plt.ylabel('Mean Square Distance')
            plt.legend()
            plt.savefig("(<r^2>).pdf")
            

    plt.show()

        
if __name__ == "__main__":
    rwalk() # make rwalk or make rwalk PART=1,2


    
