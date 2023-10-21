import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time


total_len = 20 
grid_points = 400
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
    eps = 1e-3  # Convergence threshold
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

        
                  
                    error += abs(matrix[i, j] - sum/4+(rho[i][j]*step**2)/4)
                    matrix[i, j] = sum/4+(rho[i][j]*step**2)/4
        
        
        # error = abs(np.max(matrix)-np.max(new_matrix))

        # if iterations > 5:
        #     error =  abs(np.max(matrix)-np.max(new_matrix))
     
        # error = error/float(grid_points)
       
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














"""
matrix_size = int(radius / step) + 1
matrix = np.zeros((matrix_size, matrix_size))
coor_toignore = []
data = []

k = 0
while y >= -10:
    i = 0
    for j in range(len(h)):
        if ((x + h[j])**2 + y**2 ) <=100:
            # print(x+h[j], y)
            x_list.append(x + h[j])
            y_list.append(y)
            matrix[i][k] = 0.00001
            data.append((x + h[j],y,i,j))
            
        else:
            matrix[i][k] = 0
            coor_toignore.append((i,k))
            data.append((x + h[j],y,i,j))
        i = i+1
    y = y - step
    k=k+1
    
matrix[103][100] = 1
matrix[97][100] = -1

# print(coor_toignore)
    
# Below line is for the visulisation of each points
plt.figure(1,figsize=(5, 5))
plt.scatter(x_list, y_list, marker='o', alpha=0.5, s =1)
plt.scatter(-0.3,0, color='b', s=1.5)
plt.scatter(0.3,0, color='r', s=1.5)
plt.xlim(-radius/2, radius/2)
plt.ylim(-radius/2, radius/2)
plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
# print(len(matrix))
# print(len(matrix[2]))

new_matrix = np.zeros((matrix_size,matrix_size),dtype=float)
n = 0
valid_surrounding_value = []
while True:
    for i in range(200):
        for j in range(200):
            surrounding = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
            del valid_surrounding_value [:]
            for s in surrounding:
                x,y = s
                if x <= 200 and y<=200:
                    valid_surrounding_value.append(matrix[x][y])
            
            new_matrix[i, j] = np.mean(valid_surrounding_value)
            
    sum_1 = 0
    sum_2 = 0
    
    if n > 3: ######## Ask about iteration to others
        for i in range(200):
            for j in range(200):
                if (i, j) not in coor_toignore:
                    sum_1 += matrix[i][j]
                    sum_2 += new_matrix[i][j]
        diff = abs(sum_1-sum_2)
        threshold = 0.000001
        if diff < threshold:
            print(f"It took {n} iteration for {threshold} thresold") ######## Ask about iteration to others
            break
        else:
            continue

    matrix = new_matrix
    n=n+1    

plt.figure(2)
cmap = plt.get_cmap('jet')
plt.imshow(np.transpose(new_matrix), cmap=cmap)
plt.colorbar()
plt.title('Jaccobi')
plt.xlim(80, 120)
plt.ylim(80, 120)
plt.show()

quit()
# print(data)

x_new = []
y_new =[]
xy_val = []
# print(len(data))

######3
Convert to cartesian
for k in range(len(data)):
    x = data[k][0] 
    y = data[k][1] 
    i = data[k][2] 
    j = data[k][3]
    val = new_matrix[i][j]
    x_new.append(x)
    y_new.append()
    xy_val.append(val)

# Create a scatter plot with color-mapped values
# print(len(x_new))
plt.scatter(x_new, y_new, c=xy_val, cmap='viridis',s=1)
plt.colorbar()

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Heatmap')

# Show the heatmap
plt.show()
#######

n = 0
while True:
    for i in range(200):
        for j in range(200):
            surrounding = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
            valid_surrounding_value = []
            for s in surrounding:
                x,y = s
                if x <= 200 and y<=200:
                    valid_surrounding_value.append(matrix[x][y])
            
            matrix[i, j] = np.mean(valid_surrounding_value)
            
    sum_1 = 0
    sum_2 = 0
    
    if n > 10: ######## Ask about iteration to others
        break
        # for i in range(200):
        #     for j in range(200):
        #         if (i, j) not in coor_toignore:
        #             sum_1 += matrix[i][j]
        #             sum_2 += new_matrix[i][j]
        # diff = abs(sum_1-sum_2)
        # threshold = 0.000001
        # if diff < threshold:
        #     print(f"It took {n} iteration for {threshold} thresold") ######## Ask about iteration to others
        #     break
    n=n+1    

plt.figure(3)
cmap = plt.get_cmap('jet')
plt.imshow(np.transpose(matrix), cmap=cmap)
plt.colorbar()
plt.title('SOR')
plt.xlim(80, 120)
plt.ylim(80, 120)
plt.show()
"""