import matplotlib.pyplot as plt
import math
import numpy as np

def oscillator():
    l = 9.8
    g = 9.8
    gamma = 0.25
    alpha_d = 0.2
    theta0_1 = 1
    theta0_2 = 3
    omega0 = 1
    dt = 0.005
    t = []
    theta_1 = []
    theta_2 = []
    omega_1 = []
    omega_2 = []
    delta_t = []
    t.append(0)
    theta_1.append(theta0_1)
    theta_2.append(theta0_2)
    omega_1.append(omega0)
    omega_2.append(omega0)

    i = 0
    
    #Euler cromer
    while t[i] < 100:
        omega_1.append(-(g/l*theta_1[i]*dt) -2*gamma*omega_1[i]*dt+alpha_d*np.sin(1*t[i])*dt+omega_1[i])
        
        if theta_1[i]+omega_1[i+1]*dt > math.pi:
            theta_1.append(2*math.pi - theta_1[i]+omega_1[i+1]*dt)
        elif theta_1[i]+omega_1[i+1]*dt < - math.pi:
            theta_1.append(2*math.pi + theta_1[i]+omega_1[i+1]*dt)
        else:
            theta_1.append(theta_1[i]+omega_1[i+1]*dt)
    
        t.append(t[i]+dt)
        print(omega_1[i+1],theta_1[i+1], t[i+1])

        #theta2

        omega_2.append(-(g/l*theta_2[i]*dt) -2*gamma*omega_2[i]*dt+alpha_d*np.sin(1*t[i])*dt+omega_2[i])
        
        if theta_2[i]+omega_2[i+1]*dt > math.pi:
            theta_2.append(2*math.pi - theta_2[i]+omega_2[i+1]*dt)
        elif theta_1[i]+omega_2[i+1]*dt < - math.pi:
            theta_2.append(2*math.pi + theta_2[i]+omega_2[i+1]*dt)
        else:
            theta_2.append(theta_2[i]+omega_2[i+1]*dt)

        # if i >10:
        #     delta_t.append(abs(theta_1[i]-theta_2[i]))

        i = i+1
        delta_t.append(abs(theta_1[i]-theta_2[i]))


        # if omega[i-1] == omega[i]:
        #     False
    plt.figure(1,figsize=(5, 3))
    plt.semilogx(t[1:],delta_t)
    plt.title("Delta(theta1 - theta2)")
    plt.legend()
    
    plt.figure(2,figsize=(5, 3))
    plt.plot(t,theta_1, label = "tehta1o = 1")
    plt.plot(t,theta_2, label = "tehta2o = 3")
    plt.title("Euler- Cromer (theta1 vs theta2)")
    plt.legend()

    plt.figure(3,figsize=(5, 3))
    plt.plot(t,omega_1, label = "omega1")
    plt.plot(t,omega_2, label = "omega2")
    plt.title("Euler- Cromer (omega1 - omega2)")
    plt.legend()
    

    #RK4

    t_values = []
    theta_values = []
    omega_values = []

    t_values.append(0)
    theta_values.append(theta0_1)
    omega_values.append(omega0)

    h = 0.1
    i =0

    while t_values[i] < 1000: 
        # Calculate the four RK4 increments
        k1_theta = h * omega_values[i]
        k1_omega = h * (-(g/l*theta_values[i]) -2*gamma*omega_values[i]+alpha_d*np.sin(1*t[i]))
        
        k2_theta = h * (omega_values[i] + 0.5 * k1_omega)
        k2_omega = h * (-(g / l) * theta_values[i] - 2 * gamma * omega_values[i] + alpha_d * np.sin(1 * t[i]) + 0.5 * (-g / l) * (theta_values[i] + 0.5 * k1_theta) - 2 * gamma * (omega_values[i] + 0.5 * k1_omega) + alpha_d * np.sin(0.01 * (t[i] + 0.5 * h)))

        k3_theta = h * (omega_values[i] + 0.5 * k2_omega)
        k3_omega = h * (-(g / l) * (theta_values[i] + 0.5 * k2_theta) - 2 * gamma * (omega_values[i] + 0.5 * k2_omega) + alpha_d * np.sin(1 * (t[i] + 0.5 * h)))

        k4_theta = h * (omega_values[i] + k3_omega)
        k4_omega = h * (-(g / l) * (theta_values[i] + k3_theta) - 2 * gamma * (omega_values[i] + k3_omega) + alpha_d * np.sin(1 * (t[i] + h)))

        
        # Update values
        theta_values.append((k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6.0 + theta_values[i])
        omega_values.append((k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6.0 + omega_values[i])
        t_values.append(t_values[i]+h)

        print(omega_values[i],theta_values[i])
        i=i+1

    plt.figure(4, figsize=(5, 3))
    plt.plot(t_values,theta_values, label = "tehta_rk")
    plt.plot(t_values,omega_values, label = "omega_rk")
    plt.title("Runge Kutte - 4th order")
    plt.legend()
    plt.show(block=True)


if __name__ == "__main__":
    oscillator()