# point_charge.py - Iterative solution of 2-D PDE, electrostatics

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Set dimensions of the problem
L = 1.0
N = 21
ds = L / N

# Define arrays used for plotting
x = np.linspace(0, L, N)
y = np.copy(x)
X, Y = np.meshgrid(x, y)

# Make the charge density matrix
rho0 = 1.0
rho = np.zeros((N, N))
rho[int(round(N/2.0)), int(round(N/2.0))] = rho0

# Make the initial guess for the solution matrix
V = np.zeros((N, N))

# Solver
iterations = 0
eps = 1e-8  # Convergence threshold
error = 1e4  # Large dummy error

while iterations < 1e4 and error > eps:
    V_temp = np.copy(V)
    error = 0
    print(iterations)

    for j in range(2, N - 1):
        for i in range(2, N - 1):
            V[i, j] = 0.25 * (V_temp[i + 1, j] + V_temp[i - 1, j] +
                              V_temp[i, j - 1] + V_temp[i, j + 1] + rho[i, j] * ds**2)
            error += abs(V[i, j] - V_temp[i, j])

    iterations += 1
    error /= float(N)

print("Iterations =", iterations)

# Plotting
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

CS = plt.contour(X, Y, V, 30)  # Make a contour plot
plt.clabel(CS, inline=1, fontsize=10)
plt.title('PDE solution of a point charge')

CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.show()
