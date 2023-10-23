import numpy as np



total_len = 20 
grid_points = 800
step = total_len/grid_points
print(step)
a = 0.6

# h = np.arange(0,20,step)
# print(len(h))
r = 10

x = np.linspace(r, -r, num=grid_points+1)
y = np.linspace(r, -r, num=grid_points+1)
X, Y = np.meshgrid(x, y)
matrix_size = grid_points
matrix = np.zeros((matrix_size, matrix_size))
rho = np.zeros((matrix_size, matrix_size))

x_pos_cor = int((grid_points/2)+(0.5*a/step))
x_neg_cor = int((grid_points/2)-(0.5*a/step))
y_cor = int(grid_points/2)
print(x_pos_cor,x_neg_cor,y_cor )

rho[x_pos_cor][y_cor] = 1
rho[x_neg_cor][y_cor] = -1

# print(X[103][100])
# print(X[100][100])
print(X[int(grid_points/2)])
