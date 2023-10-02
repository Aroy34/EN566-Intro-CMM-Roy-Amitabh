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
x_trajec_dds = []
y_trajec_dds = []
g = 9.81
vx = []
vy = []
vx_d = []
vy_d = []
vx_dd = []
vy_dd = []
vx_dds = []
vy_dds = []
d = 1.29
A = 0.0014
m = 0.046



def golf_trajectory():
    psr = argp.ArgumentParser("Golf_Trajectory")
    psr.add_argument('--plot', type=str, default=0,
                     help="enter the value of theta seperated by ','")
    arg = psr.parse_args()
    angle = arg.plot.split(",")
    theta = [int(angl) for angl in angle] # check this again
    plt.figure(figsize=(9, 7))

    for l in range(4):
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

                plt.plot(x_trajec[0:i-1], y_trajec[0:i-1],':', label=str(theta[k])+"°"+"(Ideal)")

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
                        vx_d[i]+(-1*0.5*d*A*math.sqrt(vx_d[i]**2+vy_d[i]**2)*vx_d[i]*dt)/m)
                    y_trajec_d.append(y_trajec_d[i]+vy_d[i]*dt)
                    vy_d.append(vy_d[i]-g*dt+(-1*0.5*d*A*math.sqrt(vx_d[i]**2+vy_d[i]**2)*vx_d[i]*dt)/m)

                    if y_trajec_d[i+1] <= 0:
                        break
                    i = i+1
                
                plt.plot(x_trajec_d[0:i-1], y_trajec_d[0:i-1],'-.',
                         label=str(theta[k])+"°"+"(Drag)")
                # print(y_trajec_d[i],y_trajec_d[i-1],y_trajec_d[i-2])


            
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
                    y_trajec_dd.append(y_trajec_dd[i]+vy_dd[i]*dt)
                    if math.sqrt(vx_dd[i]**2+vy_dd[i]**2) >14:
                        vx_dd.append(vx_dd[i]+(-1*7*d*A*vx_dd[i]*dt)/m)
                        vy_dd.append(vy_dd[i]-g*dt+(-1*7*d*A*vy_dd[i]*dt)/m)
                    else: 
                        vx_dd.append(vx_dd[i]+(-1*0.5*d*A*math.sqrt(vx_dd[i]**2+vy_dd[i]**2)*vx_dd[i]*dt)/m)
                        vy_dd.append(vy_dd[i]-g*dt+(-1*0.5*d*A*math.sqrt(vx_dd[i]**2+vy_dd[i]**2)*vy_dd[i]*dt)/m)
                    
                    if y_trajec_dd[i+1] <= 0:
                        break
                    i = i+1

                plt.plot(x_trajec_dd[0:i-1], y_trajec_dd[0:i-1],'--',
                         label=str(theta[k])+"°"+"(Dimpled + Drag)")
                # plt.annotate(f'({x_trajec_dd[i-1]:.2f}, {y_trajec_dd[i-1]:.2f})', (x_trajec_dd[i-1], y_trajec_dd[i-1]), fontsize=3,xytext=(-5, 5), textcoords='offset points',
                #  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5'))
                # print(y_trajec_dd[i],y_trajec_dd[i-1],y_trajec_dd[i-2])

            if l == 3:
                del x_trajec_dds[:]
                del y_trajec_dds[:]
                del vx_dds[:]
                del vy_dd[:]
                x_trajec_dds.append(0)
                y_trajec_dds.append(0)
                vx_dds.append(v*math.cos(math.radians(theta[k])))
                vy_dds.append(v*math.sin(math.radians(theta[k])))
                i = 0
                while True:
                    x_trajec_dds.append(x_trajec_dds[i]+vx_dds[i]*dt)
                    y_trajec_dds.append(y_trajec_dds[i]+vy_dds[i]*dt)
                    if math.sqrt(vx_dds[i]**2+vy_dds[i]**2) >14:
                        vx_dds.append(vx_dds[i]+(-1*7*d*A*vx_dds[i]*dt)/m-0.25*vy_dds[i]*dt)
                        vy_dds.append(vy_dds[i]-g*dt+(-1*7*d*A*vy_dds[i]*dt)/m+0.25*vx_dds[i]*dt)
                    else: 
                        vx_dds.append(vx_dds[i]+(-1*0.5*d*A*math.sqrt(vx_dds[i]**2+vy_dds[i]**2)*vx_dds[i]*dt)/m-0.25*vy_dds[i]*dt*dt)
                        vy_dds.append(vy_dds[i]-g*dt+(-1*0.5*d*A*vy_dds[i]*dt)/m+0.25*vx_dds[i]*dt)
        
        
                    if y_trajec_dds[i+1] <= 0:
                        break
                    i = i+1


                plt.plot(x_trajec_dds[0:i-1], y_trajec_dds[0:i-1],'-',
                         label=str(theta[k])+"°"+"(Spin + Dimpled + Drag)")


    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.legend(loc='best')
    plt.title("Trajectory of golf ball")
    plt.savefig("Golf_Trajectory.jpeg")
    plt.show()
    
if __name__ == "__main__":
    golf_trajectory()
