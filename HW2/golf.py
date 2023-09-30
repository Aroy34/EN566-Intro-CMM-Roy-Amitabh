import numpy as np
import matplotlib.pyplot as plt
import math
import argparse as argp



v = 40
dt = 0.005
x_trajec = []
y_trajec = []
x_trajec_d = []
y_trajec_d = []
x_trajec_dd = []
y_trajec_dd = []
g = 9.81
vx = []
vy = []
vx_d = []
vy_d = []
vx_dd = []
vy_dd = []
c = 0.5
c_dim = 7/v
d = 1.29
A = 0.0014
m = 0.046
Fdrag = -1*c*d*A*v**2
Fdrag_dim = -1*c_dim*d*A*v**2


def golf_trajectory():
    psr = argp.ArgumentParser("Golf_Trajectory")
    psr.add_argument('--plot', type=str, default=0,
                     help="enter the value of theta seperated by ','")
    arg = psr.parse_args()
    angle = arg.plot.split(",")
    theta = [int(angl) for angl in angle] # check this again

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
                i = 0
                while True:
                    x_trajec.append(x_trajec[i]+vx[i]*dt)
                    vx.append(vx[i])

                    y_trajec.append(y_trajec[i]+vy[i]*dt)
                    vy.append(vy[i]-g*dt)

                    if y_trajec[i+1] <= 0:
                        break

                    i = i+1

                plt.plot(x_trajec, y_trajec, label=str(theta[k])+"°"+"(Ideal)")

            if l == 1:
                
                del x_trajec_d[:]
                del y_trajec_d[:]
                del vx_d[:]
                del vy_d[:]
                x_trajec_d.append(0)
                y_trajec_d.append(0)
                vx_d.append(v*math.cos(math.radians(theta[k])))
                vy_d.append(v*math.sin(math.radians(theta[k])))
                i = 0
                while True:
                    x_trajec_d.append(x_trajec_d[i]+vx_d[i]*dt)
                    vx_d.append(
                        vx_d[i]+(Fdrag*math.cos(math.radians(theta[k]))*dt)/m)
                    y_trajec_d.append(y_trajec_d[i]+vy_d[i]*dt)
                    vy_d.append(vy_d[i]-g*dt+Fdrag*math.sin(math.radians(theta[k]))*dt/m)

                    if y_trajec_d[i+1] <= 0:
                        break
                    i = i+1
                
                plt.plot(x_trajec_d[0:i-1], y_trajec_d[0:i-1],
                         label=str(theta[k])+"°"+"W/ Drag")

            
            if l == 2:
                del x_trajec_dd[:]
                del y_trajec_dd[:]
                del vx_dd[:]
                del vy_dd[:]
                x_trajec_dd.append(0)
                y_trajec_dd.append(0)
                vx_dd.append(v*math.cos(math.radians(theta[k])))
                vy_dd.append(v*math.sin(math.radians(theta[k])))
                i = 0
                while True:
                    x_trajec_dd.append(x_trajec_dd[i]+vx_dd[i]*dt)
                    vx_dd.append(
                        vx_dd[i]+(Fdrag_dim*math.cos(math.radians(theta[k]))*dt)/m)
                
                    y_trajec_dd.append(y_trajec_dd[i]+vy_dd[i]*dt)
                    vy_dd.append(vy_dd[i]-g*dt+Fdrag_dim*math.sin(math.radians(theta[k]))*dt/m)
                    if y_trajec_dd[i+1] <= 0:
                        break
                    i = i+1

                plt.plot(x_trajec_dd[0:i-1], y_trajec_dd[0:i-1],
                         label=str(theta[k])+"°"+"Dimpled + Drag")

    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.title("Trajectory of golf ball")
    plt.show()


if __name__ == "__main__":
    golf_trajectory()
