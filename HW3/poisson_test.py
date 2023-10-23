import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse as argp


def poisson():
    """The function can do claculations using jaccobi and 
    Simultaneous Over-Relaxation methods for an electric dipole"""
    psr = argp.ArgumentParser("Poisson")
    psr.add_argument('--part', type=str, default=0,
                     help="enter the part ','")  # python poisson.py --part 1,2,3
    arg = psr.parse_args()
    part_str = arg.part.split(",")
    part_list = [int(pt) for pt in part_str]  # list fo all the step widths

    total_len = 20 # Distance between two extreams
    a = 0.6 # Distance between dipoles
    r = 10 # Spherical boundary conditions

    def jacobi(tolerance_list, grid_pts):
        """
        Function takes tollerance values and grid points to do the calculations

        Returns: 
        grid_points : Grid point generated
        tol : Tolerance taken
        iteration : Total number of iterations
        """

        return_msg = []
        for grid_points in grid_pts:
            step = total_len/grid_points

            x = np.linspace(r, -r, num=grid_points+1)
            y = np.linspace(r, -r, num=grid_points+1)
            X, Y = np.meshgrid(x, y) # 2D Array for x,y coordinats
            matrix_size = grid_points+1
            matrix = np.zeros((matrix_size, matrix_size))
            rho = np.zeros((matrix_size, matrix_size))

            # Coordinated for the dipoles
            x_pos_cor = int((grid_points/2)+(0.5*a/step))
            x_neg_cor = int((grid_points/2)-(0.5*a/step))
            y_cor = int(grid_points/2)

            # Charge of each dipole
            rho[x_pos_cor][y_cor] = 1
            rho[x_neg_cor][y_cor] = -1

            new_matrix = np.zeros((matrix_size, matrix_size))

            for tol in tolerance_list:
                print(tol)
                iterations = 0
                error = 1e4
                diff = []

                while iterations < 10000 and error > tol:
                    del diff[:]
                    print(iterations)
                    new_matrix = np.copy(matrix)

                    for j in range(grid_points):
                        for i in range(grid_points):
                            if X[i][j]**2+Y[i][j] < r**2:
                                surrounding = [
                                    (i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                                sum = 0
                                # valid_surrounding_value.append(rho[i][j]*step**2)
                                for s in surrounding:
                                    x, y = s
                                    if x < grid_points and y < grid_points:
                                        if X[x][y]**2+Y[x][y]**2 < 100:
                                            sum = sum + new_matrix[x][y]

                                if iterations > 5:
                                    diff.append(
                                        abs((sum/4+(rho[i][j]*step**2)/4) - new_matrix[i][j]))

                                matrix[i][j] = sum/4+(rho[i][j]*step**2)/4

                    if iterations > 5:
                        error = np.sum(diff)
                    print(error)
                    iterations += 1

                return_msg.append((grid_points, tol, iterations))

                # Ref 4 for the ploting
                plt.figure(figsize=(10, 8))
                matplotlib.rcParams['xtick.direction'] = 'out'
                matplotlib.rcParams['ytick.direction'] = 'out'
                CS = plt.contour(X, Y, np.transpose(
                    matrix), 30, label=f"(Step = {step}, Tolerance = {tol}, Iterations = {iterations} )")  # Make a contour plot
                plt.clabel(CS, inline=1, fontsize=10)
                plt.title(
                    f"Jaccobi: Electric Potential of a Static Electric Dipole({step}_{grid_points}_{tol}_{iterations})")
                CB = plt.colorbar(CS, shrink=0.8, extend='both')
                plt.legend()
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.xlabel('X-Axis')
                plt.ylabel('Y-Axis')
                plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
                plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
                plt.savefig(
                    f"Jaccobi: Electric Potential of a Static Electric Dipole({step}_{grid_points}_{tol}_{iterations}).pdf")
                # plt.show()

        return return_msg

    def sor(fixed_accy, grid_pts, omega):
        """
        Function takes fixed accuracy, grid points and
        relaxation parameter to do the calculations

        Returns: 
        grid_points : Grid point generated
        Fixed accuracy : fixed accurracy taken
        iteration : Total number of iterations
        """
        return_msg = []
        for grid_points in grid_pts:
            step = total_len/grid_points

            x = np.linspace(r, -r, num=grid_points+1)
            y = np.linspace(r, -r, num=grid_points+1)
            X, Y = np.meshgrid(x, y) # 2D Array for x,y coordinates
            matrix_size = grid_points+1
            matrix_2 = np.zeros((matrix_size, matrix_size))
            rho = np.zeros((matrix_size, matrix_size))

            # Coordinated for the dipoles
            x_pos_cor = int((grid_points/2)+(0.5*a/step))
            x_neg_cor = int((grid_points/2)-(0.5*a/step))
            y_cor = int(grid_points/2)

            # Charge of each dipole
            rho[x_pos_cor][y_cor] = 1
            rho[x_neg_cor][y_cor] = -1

            for fixed_accuracy in fixed_accy:

                iterations = 0
                rel_change = 1000
                diff = []

                while iterations < 1000 and rel_change > fixed_accuracy:

                    print(iterations)
                    del diff[:]

                    for j in range(grid_points):
                        for i in range(grid_points):
                            sum = 0
                            if X[i][j]**2+Y[i][j] < r**2:
                                surrounding = [
                                    (i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                                for s in surrounding:
                                    x, y = s
                                    if x < grid_points and y < grid_points:
                                        if X[x][y]**2+Y[x][y]**2 <= r**2:
                                            sum = sum + matrix_2[x][y]

                            new = (1-omega)*matrix_2[i][j]+omega * \
                                sum/4+omega*(rho[i][j]*step**2)/4
                            diff.append(abs(new - matrix_2[i][j]))
                            matrix_2[i][j] = new

                    rel_change = max(diff)
                    print(rel_change)
                    iterations += 1

                return_msg.append((grid_points, fixed_accuracy, iterations))

                # Ref 4 for the ploting
                plt.figure(figsize=(10, 8))
                matplotlib.rcParams['xtick.direction'] = 'out'
                matplotlib.rcParams['ytick.direction'] = 'out'
                CS = plt.contour(X, Y, np.transpose(
                    matrix_2), 30, label=f"(Step = {step}, Fixed accuracy = {fixed_accuracy}, Iterations = {iterations} )")  # Make a contour plot
                plt.clabel(CS, inline=1, fontsize=10)
                plt.title(
                    f"SOR: Electric Potential of a Static Electric Dipole({step}_{grid_points}_{fixed_accuracy}_{iterations})")
                CB = plt.colorbar(CS, shrink=0.8, extend='both')
                plt.legend()
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.xlabel('X-Axis')
                plt.ylabel('Y-Axis')
                plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
                plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
                plt.savefig(
                    f"SOR: Electric Potential of a Static Electric Dipole ({step}_{grid_points}_{fixed_accuracy}_{iterations}).pdf")
                # plt.show()
        return return_msg

    for i in range(len(part_list)):
        if part_list[i] == 1:
            grid_num = [400]
            tolerance = [0.0001]
            msg = jacobi(tolerance, grid_num)
            for i in range(len(msg)):
                print(
                    f"Jaccobi: For {msg[i][0]} grid points, it took {msg[i][2]} iteration for tolerance (error) = {msg[i][1]}")
        elif part_list[i] == 2:
            grid_num = [400]
            tolerance = np.linspace(0.001, 0.0001, 3).tolist()
            msg = jacobi(tolerance, grid_num)
            for i in range(len(msg)):
                print(
                    f"Jaccobi: For {msg[i][0]} grid points, it took {msg[i][2]} iteration for tolerance (error) = {msg[i][1]}")
        elif part_list[i] == 3:
            grid_num = [200, 400, 800]
            fixed_acrcy = [1e-6]
            omega = 1.2
            msg = sor(fixed_acrcy, grid_num, omega)
            for i in range(len(msg)):
                print(
                    f"SOR: For {msg[i][0]} grid points, it took {msg[i][2]} iteration for fixed accuracy = {msg[i][1]}")


if __name__ == "__main__":
    poisson() # Calling the function (make oscillator PART=1,2,3)


