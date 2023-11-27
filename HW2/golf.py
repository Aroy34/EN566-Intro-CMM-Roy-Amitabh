# Import matplotlib.pyplot for plotting the graph
# Import math for doing mathermatical operations
# Import argparse for passing arguments from the terminal to python code
import matplotlib.pyplot as plt
import math
import argparse as argp

v = 40  # Initial velocity,m/s
dt = 0.005  # step size,s
x_trajec = []  # Ideal x-axis trajectory
y_trajec = []  # Ideal y-axis trajectory
x_trajec_d = []  # drag x-axis trajectory
y_trajec_d = []  # drag y-axis trajectory
x_trajec_dd = []  # dimpled+spin x-axis trajectory
y_trajec_dd = []  # dimpled+spin y-axis trajectory
x_trajec_dds = []  # drag+dimpled+spin x-axis trajectory
y_trajec_dds = []  # drag+dimpled+spin y-axis trajectory
g = 9.81  # Accelarationn due to gravity
vx = []  # Ideal x-axis velocity
vy = []  # Ideal y-axis velocity
vx_d = []  # drag x-axis velocity
vy_d = []  # drag y-axis velocity
vx_dd = []  # dimpled+spin x-axis velocity
vy_dd = []  # dimpled+spin y-axis velocity
vx_dds = []  # drag+dimpled+spin x-axis velocity
vy_dds = []  # drag+dimpled+spin y-axis velocity
d = 1.29  # Density of the air, kg/m^3
A = 0.0014  # Area of the ball, m^2
m = 0.046  # Mass of the ball, kg


def golf_trajectory():
    """ This functions take argument --plot=angle.The code calculate
    the trajectory of a golf ball and plot it for the conditions like
    ideal, smooth ball with drag, dimpled ball with drag and dimpled ball
    with drag + spin """

    psr = argp.ArgumentParser("Golf_Trajectory")
    psr.add_argument('--plot', type=str, default=0,
                     help="enter the value of theta seperated by ','")
    arg = psr.parse_args()
    angle = arg.plot.split(",")
    theta = [int(angl) for angl in angle]  # list of the theta values
    plt.figure(figsize=(9, 7))
    '''For loop for cyclying though each conditons
    and through all the given angles'''
    for condition in range(3,4):
        for k in range(len(theta)):
            if condition == 0:
                # Emptying the list
                del x_trajec[:]
                del y_trajec[:]
                del vx[:]
                del vy[:]
                x_trajec.append(0)  # Adding the t=o value
                y_trajec.append(0)  # Adding the t=o value
                # Adding the t=o value
                vx.append(v*math.cos(math.radians(theta[k])))
                # Adding the t=o value
                vy.append(v*math.sin(math.radians(theta[k])))
                i = 0
                while True:
                    # Appeding the next value
                    x_trajec.append(x_trajec[i]+vx[i]*dt)
                    vx.append(vx[i])  # Appeding the next value
                    # Appeding the next value
                    y_trajec.append(y_trajec[i]+vy[i]*dt)
                    vy.append(vy[i]-g*dt)  # Appeding the next value
                    
                    # Checking if the ball has touched the ground or not
                    if y_trajec[i+1] <= 0:
                    
                        break

                    i = i+1

                plt.plot(x_trajec[0:i-1], y_trajec[0:i-1],
                         ':', label=str(theta[k])+"°"+"(Ideal)")

            if condition == 1:
                # Emptying the list
                del x_trajec_d[:]
                del y_trajec_d[:]
                del vx_d[:]
                del vy_d[:]
                x_trajec_d.append(0)  # Adding the t=o value
                y_trajec_d.append(0)  # Adding the t=o value
                # Adding the t=o value
                vx_d.append(v*math.cos(math.radians(theta[k])))
                # Adding the t=o value
                vy_d.append(v*math.sin(math.radians(theta[k])))
                i = 0
                while True:
                    # Appeding the next value
                    x_trajec_d.append(x_trajec_d[i]+vx_d[i]*dt)
                    vx_d.append(
                        vx_d[i]+(-1*0.5*d*A*math.sqrt(vx_d[i]**2+vy_d[i]**2)
                                 *vx_d[i]*dt)/m)  # Appeding the next value
                    # Appeding the next value
                    y_trajec_d.append(y_trajec_d[i]+vy_d[i]*dt)
                    vy_d.append(
                        vy_d[i]-g*dt+(-1*0.5*d*A*math.sqrt(vx_d[i]**2+vy_d[i]**2)
                                      *vx_d[i]*dt)/m)  # Appeding the next value

                    # Checking if the ball has touched the ground or not
                    if y_trajec_d[i+1] <= 0:
                        break
                    i = i+1

                plt.plot(x_trajec_d[0:i-1], y_trajec_d[0:i-1], '-.',
                         label=str(theta[k])+"°"+"(Drag)")

            if condition == 2:
                # Emptying the list
                del x_trajec_dd[:]
                del y_trajec_dd[:]
                del vx_dd[:]
                del vy_dd[:]
                x_trajec_dd.append(0)  # Adding the t=o value
                y_trajec_dd.append(0)  # Adding the t=o value
                # Adding the t=o value
                vx_dd.append(v*math.cos(math.radians(theta[k])))
                # Adding the t=o value
                vy_dd.append(v*math.sin(math.radians(theta[k])))
                i = 0
                while True:
                    # Appeding the next value
                    x_trajec_dd.append(x_trajec_dd[i]+vx_dd[i]*dt)
                    # Appeding the next value
                    y_trajec_dd.append(y_trajec_dd[i]+vy_dd[i]*dt)
                    # Checking for the condition to select optimum C value
                    if math.sqrt(vx_dd[i]**2+vy_dd[i]**2) > 14:
                        # Appeding the next value
                        vx_dd.append(vx_dd[i]+(-1*7*d*A*vx_dd[i]*dt)/m)
                        # Appeding the next value
                        vy_dd.append(vy_dd[i]-g*dt+(-1*7*d*A*vy_dd[i]*dt)/m)
                    else:
                        vx_dd.append(
                            vx_dd[i]+(-1*0.5*d*A*math.sqrt(vx_dd[i]**2+vy_dd[i]**2)
                                      * vx_dd[i]*dt)/m)  # Appeding the next value
                        vy_dd.append(
                            vy_dd[i]-g*dt+(-1*0.5*d*A*math.sqrt(vx_dd[i]**2+vy_dd[i]**2)
                                           * vy_dd[i]*dt)/m)  # Appeding the next value

                    # Checking if the ball has touched the ground or not
                    if y_trajec_dd[i+1] <= 0:
                        break
                    i = i+1

                plt.plot(x_trajec_dd[0:i-1], y_trajec_dd[0:i-1], '--',
                         label=str(theta[k])+"°"+"(Dimpled + Drag)")

            if condition == 3:
                # Emptying the list
                del x_trajec_dds[:]
                del y_trajec_dds[:]
                del vx_dds[:]
                del vy_dd[:]
                x_trajec_dds.append(0)  # Adding the t=o value
                y_trajec_dds.append(0)  # Adding the t=o value
                # Adding the t=o value
                vx_dds.append(v*math.cos(math.radians(theta[k])))
                # Adding the t=o value
                vy_dds.append(v*math.sin(math.radians(theta[k])))
                i = 0
                while True:
                    # Appeding the next value
                    x_trajec_dds.append(x_trajec_dds[i]+vx_dds[i]*dt)
                    # Appeding the next value
                    y_trajec_dds.append(y_trajec_dds[i]+vy_dds[i]*dt)
                    # Checking for the condition to select optimum C value
                    if math.sqrt(vx_dds[i]**2+vy_dds[i]**2) > 14:
                        vx_dds.append(
                            vx_dds[i]+(-1*7*d*A*vx_dds[i]*dt)/m-0.25
                            * vy_dds[i]*dt)  # Appeding the next value
                        vy_dds.append(
                            vy_dds[i]-g*dt+(-1*7*d*A*vy_dds[i]*dt)/m+0.25
                            * vx_dds[i]*dt)  # Appeding the next value
                    else:
                        vx_dds.append(
                            vx_dds[i]+(-1*0.5*d*A*math.sqrt(vx_dds[i]**2+vy_dds[i]**2)
                                       * vx_dds[i]*dt)/m-0.25*vy_dds[i]*dt)  # Appeding the next value
                        vy_dds.append(
                            vy_dds[i]-g*dt+(-1*0.5*d*A*vy_dds[i]*dt)/m+0.25
                            * vx_dds[i]*dt)  # Appeding the next value

                    # Checking if the ball has touched the ground or not
                    if y_trajec_dds[i+1] <= 0:
                        break
                    i = i+1

                plt.plot(x_trajec_dds[0:i-1], y_trajec_dds[0:i-1], '-',
                         label=str(theta[k])+"°"+"(Spin + Dimpled + Drag)")

    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.legend(loc='best')
    plt.title("Trajectory of golf ball")
    plt.savefig("Golf_Trajectory.jpeg")
    plt.show()


if __name__ == "__main__":
    golf_trajectory()  # calling the function (make golf THETA=45,35,15,9)
