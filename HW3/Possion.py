import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time


total_len = 20 
grid_points = 200
step = total_len/grid_points
a = 0.6

# h = np.arange(0,20,step)
# print(len(h))
r = 10
negative = -1
positive = 1
surrounding = 0.1
x_list = []
y_list = []
k = 0

x = np.linspace(r, -r, num=grid_points)
y = np.linspace(r, -r, num=grid_points)
X, Y = np.meshgrid(x, y)
matrix_size = grid_points
matrix = np.zeros((matrix_size, matrix_size))
rho = np.zeros((matrix_size, matrix_size))

x_pos_cor = int((grid_points/2)+(0.3/step))
x_neg_cor = int((grid_points/2)-(0.3/step))
y_cor = int(grid_points/2)

rho[x_pos_cor][y_cor] = 1
rho[x_neg_cor][y_cor] = -1

print(len(Y[1]))
new_matrix = np.zeros((matrix_size, matrix_size))
matrix_2 = np.zeros((matrix_size, matrix_size))

coor_toignore = []

for i in range(len(matrix)):
    for j in range(len(matrix)):
        if ((X[i][j])**2 + (Y[i][j])**2 ) >= 100:
            coor_toignore.append((i,j))

def jaccobi():
    iterations = 0
    eps = 1e-2  # Convergence threshold
    error = 1e4  # Large dummy error

    while iterations < 10000 and error > eps:
        if iterations >5:
            error = 0
        print(iterations)
        new_matrix = np.copy(matrix)

        for j in range(grid_points):
            for i in range(grid_points):
                if X[i][j]**2+Y[i][j] <r**2:
                    surrounding = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
                    sum = 0
                    # valid_surrounding_value.append(rho[i][j]*step**2)
                    for s in surrounding:
                        x,y = s
                        if x < grid_points and y<grid_points:
                            if X[x][y]**2+Y[x][y]**2 <100:
                                sum = sum + new_matrix[x][y]

                    error += round(abs(matrix[i, j] - sum/4+(rho[i][j]*step**2)/4),10)
                    matrix[i, j] = sum/4+(rho[i][j]*step**2)/4
       
        print(error)
        iterations += 1

    # contours = plt.contour(X, Y, np.transpose(matrix), colors='k')
    # plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(np.transpose(new_matrix), extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto', origin='lower')
    plt.colorbar()
    # Label the axes
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    # Show the plot
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Electric Potential of a Static Electric Dipole - Jaccobi (Step = )" ,step,")")
    plt.show()

def sor():
    iterations = 0
    eps = 1e-3  # Convergence threshold
    error = 1e4  # Large dummy error

    while iterations < 1000 and error > eps:
        if iterations >5:
            error = 0
        print(iterations)

        for j in range(200):
            for i in range(200):
                if X[i][j]**2+Y[i][j] <100:
                    surrounding = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
                    sum = 0
                    for s in surrounding:
                        x,y = s
                        if x < 200 and y<200:
                            if X[x][y]**2+Y[x][y]**2 <=100:
                                sum = sum + matrix_2[x][y]
                    
                if iterations > 5:
                    error +=(abs(sum/4+(rho[i][j]*step**2)/4 - matrix_2[i, j]))    
                matrix_2[i, j] = sum/4+(rho[i][j]*step**2)/4

        if iterations > 5:
            # error /= float(matrix_size)
            print(error)
        iterations += 1

    # contours = plt.contour(X, Y, np.transpose(matrix), colors='b')
    # plt.clabel(contours,inline=False, fontsize=8)
    plt.imshow(np.transpose(matrix_2), extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto', cmap='jet', origin='lower')
    plt.colorbar()
    # Label the axes
    plt.xlabel('X-Axis_SOR')
    plt.ylabel('Y-Axis')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)

    # Show the plot
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

if __name__ == "__main__":
    jaccobi()
    # sor()