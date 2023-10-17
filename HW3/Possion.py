import numpy as np
import matplotlib.pyplot as plt
import time

step = 0.1
radius = 20 
h = np.arange(0,20,step)
# print(h)
x = -10
y = 10
negative = -1
positive = 1
surrounding = 0.1
x_list = []
y_list = []
k = 0

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
            matrix[i][k] = 0
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
plt.figure(figsize=(5, 5))
plt.scatter(x_list, y_list, marker='o', alpha=0.5, s =1)
plt.scatter(-0.3,0, color='b', s=1.5)
plt.scatter(0.3,0, color='r', s=1.5)
plt.xlim(-radius/2, radius/2)
plt.ylim(-radius/2, radius/2)
plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
plt.show()
print(len(matrix))
print(len(matrix[2]))

new_matrix = np.zeros((matrix_size,matrix_size),dtype=float)
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
            
            new_matrix[i, j] = np.mean(valid_surrounding_value)
            
    sum_1 = 0
    sum_2 = 0
    
    if n > 10: ######## Ask about iteration to others
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

    matrix = new_matrix
    n=n+1    

cmap = plt.get_cmap('cividis')
plt.imshow(np.transpose(new_matrix), cmap=cmap)
plt.colorbar()
plt.title('Jaccobi')
plt.show()
plt.savefig("contour.jpeg")
# print(data)

x_new = []
y_new =[]
xy_val = []
# print(len(data))

""" Convert to cartesian
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
"""

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

cmap = plt.get_cmap('cividis')
plt.imshow(np.transpose(matrix), cmap=cmap)
plt.colorbar()
plt.title('SOR')
plt.show()