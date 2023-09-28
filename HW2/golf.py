import numpy as np
import matplotlib.pyplot as plt
import math
import argparse as argp 



'''psr = argp.ArgumentParser("Projectile Trajectory")
psr.add_argument('--plot', type=str, default=0,
                 help="enter the angle seperated by ','")
arg = psr.parse_args()'''

v = 70
dt = 0.1
x_trajec = []
y_trajec =[]
x_trajec_d = []
y_trajec_d =[]
theta = [34,45,9]
g = 9.81
vx =[]
vy = []
vx_d =[]
vy_d = []
c = 0.5
d = 1.29
A = 0.0014
m = 0.046
Fdrag = -1*c*d*A*v**2
 

for l in range(3):
    for k in range(len(theta)):
        if l == 0:
            del x_trajec[:]
            del y_trajec[:]
            del vx[:]
            del vy[:]
            x_trajec.append(0)
            y_trajec.append(0)
            vx.append(v*math.cos(math.radians(theta[k])))
            vy.append(v*math.sin(math.radians(theta[k])))
            i=0
            while True:
                x_trajec.append(x_trajec[i]+vx[i]*dt)
                vx.append(vx[i])

                y_trajec.append(y_trajec[i]+vy[i]*dt)
                vy.append(vy[i]-g*dt)

                if y_trajec[i+1] <= 0:
                    break

                i=i+1
            plt.plot(x_trajec, y_trajec,label = str(theta[k])+"°")
            plt.xlabel("Distance (m)")
            plt.ylabel("Height (m)")
            plt.legend()
            plt.title("Trajectory")
                  
        if l == 1:
            del x_trajec_d[:]
            del y_trajec_d[:]
            del vx_d[:]
            del vy_d[:]
            x_trajec_d.append(0)
            y_trajec_d.append(0)
            vx_d.append(v*math.cos(math.radians(theta[k])))
            vy_d.append(v*math.sin(math.radians(theta[k])))
            i=0
            while True:    
                x_trajec_d.append(x_trajec_d[i]+vx_d[i]*dt)
                print(x_trajec_d)
                vx_d.append(vx_d[i]-(A*math.sqrt(vx_d[i]+vy_d[i])*vx_d[i]*dt)/m)
                y_trajec_d.append(y_trajec_d[i]+vy_d[i]*dt)
                vy_d.append(vy_d[i]-g*dt-A*math.sqrt(vx_d[i]+vy_d[i])*vx_d[i]*dt/m)
                print(vy_d)

                if y_trajec_d[i+1] <= 0:
                    break

                i=i+1

            plt.plot(x_trajec_d, y_trajec_d,label = str(theta[k])+"°"+"W/ Drag")
            plt.xlabel("Distance (m)")
            plt.ylabel("Height (m)")
            plt.legend()
            plt.title("Trajectory with drag")

plt.show()


