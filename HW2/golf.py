import numpy as np
import matplotlib.pyplot as plt
import math

v = 70
theta = [45, 30, 15, 9]
dt = 0.1

x_trajec = []
y_trajec =[]

x_trajec.append(0)
y_trajec.append(0)



g = 9.81

i = 0
vx =[]
vy = []
vx.append(v*math.cos(math.radians(theta[1])))
vy.append(v*math.cos(math.radians(theta[1])))

print(vx[0],vy[0])

for k in range(len(theta)):

    while True:
        x_trajec.append(x_trajec[i]+vx[i]*dt)
        vx.append(vx[i])

        y_trajec.append(y_trajec[i]+vy[i]*dt)
        vy.append(vy[i]-g*dt)

        print(y_trajec)

        if y_trajec[i+1] <= 0:
            break

        i=i+1

plt.plot(x_trajec, y_trajec)
plt.show()

