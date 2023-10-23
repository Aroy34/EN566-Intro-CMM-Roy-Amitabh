import matplotlib.pyplot as plt
import numpy as np
import math
import argparse as argp


psr = argp.ArgumentParser("Oscilatory")
psr.add_argument('--part', type=str, default=0,
                    help="enter the part ','") #python Oscillator.py --part 1,2,3
arg = psr.parse_args()
part_str = arg.part.split(",")
part_list = [int(pt) for pt in part_str]  # list fo all the step widths

print(part_list)



def oscillator():
    l = 9.8
    g = 9.8
    gamma = 0.25
    alpha_d = 0.2
    theta0 = 0
    omega0 = 0
    omega_d = 0.8
    dt = 0.05

     # Euler cromer
    def euler_cromer(theta0=0,omeg0=0,omegd=0.8, effect = "Linear"):
        theta0_1 = theta0
        omega0 = omeg0
        omega_d = omegd
        # if effect == "non-linear":
        t = []
        theta_1 = []
        omega_1 = []
        
        t.append(0)
        theta_1.append(theta0_1)
        omega_1.append(omega0)

        if effect == "Non-Linear":
            i = 0
            while t[i] < 100:
                omega_1.append(-(g/l*np.sin(theta_1[i])*dt) - 2*gamma*omega_1[i]
                            * dt+alpha_d*np.sin(omega_d*t[i])*dt+omega_1[i])
                # print(omega_1[i],omega_1[i+1])
                theta_1.append(theta_1[i]+omega_1[i+1]*dt)
                print(theta_1[i],theta_1[i+1])
                t.append(t[i]+dt)
                # print(omega_1[i+1],theta_1[i+1], t[i+1])

                i = i+1

        else:   
            i = 0
            while t[i] < 100:
                omega_1.append(-(g/l*(theta_1[i])*dt) - 2*gamma*omega_1[i]
                            * dt+alpha_d*np.sin(omega_d*t[i])*dt+omega_1[i])
                # print(omega_1[i],omega_1[i+1])
                theta_1.append(theta_1[i]+omega_1[i+1]*dt)
                # print(theta_1[i],theta_1[i+1])
                t.append(t[i]+dt)
                # print(omega_1[i+1],theta_1[i+1], t[i+1])

                i = i+1
        return theta_1,omega_1,t

    # RK4
    def rk4(theta0=0,omeg0=0,omegd=0.8,effect = "Linear"):
        theta0_1 = theta0
        omega0 = omeg0
        omega_d = omegd

        t_values = []
        theta_values = []
        omega_values = []

        del t_values [:]
        del theta_values [:]
        del omega_values [:]

        t_values.append(0)
        theta_values.append(theta0_1)
        omega_values.append(omega0)

        if effect == "Non-Linear":
            i = 0
            while t_values[i] < 100:
                # Calculate the four RK4 increments
                k1_theta = dt * omega_values[i]
                k1_omega = dt * \
                    (-(g/l*np.sin(theta_values[i])) - 2*gamma *
                    omega_values[i]+alpha_d*np.sin(omega_d*t_values[i]))

                k2_theta = dt * (omega_values[i] + 0.5 * k1_omega)
                k2_omega = dt * (-(g / l) * np.sin(theta_values[i]) - 2 * gamma * omega_values[i] + alpha_d * np.sin(omega_d * t_values[i]) + 0.5 * (-g / l) * (
                    np.sin(theta_values[i]) + 0.5 * k1_theta) - 2 * gamma * (omega_values[i] + 0.5 * k1_omega) + alpha_d * np.sin(omega_d * (t_values[i] + 0.5 * dt)))

                k3_theta = dt * (omega_values[i] + 0.5 * k2_omega)
                k3_omega = dt * (-(g / l) * (np.sin(theta_values[i]) + 0.5 * k2_theta) - 2 * gamma * (
                    omega_values[i] + 0.5 * k2_omega) + alpha_d * np.sin(omega_d * (t_values[i] + 0.5 * dt)))

                k4_theta = dt * (omega_values[i] + k3_omega)
                k4_omega = dt * (-(g / l) * (np.sin(theta_values[i]) + k3_theta) - 2 * gamma * (
                    omega_values[i] + k3_omega) + alpha_d * np.sin(omega_d * (t_values[i] + dt)))

                # Update values
                theta_values.append((k1_theta + 2 * k2_theta + 2 *
                                    k3_theta + k4_theta) / 6.0 + theta_values[i])
                omega_values.append((k1_omega + 2 * k2_omega + 2 *
                                    k3_omega + k4_omega) / 6.0 + omega_values[i])
                t_values.append(t_values[i]+dt)

                # print(omega_values[i],theta_values[i])
                i = i+1
            

        else: 
            i = 0
            while t_values[i] < 100:
                # Calculate the four RK4 increments
                k1_theta = dt * omega_values[i]
                k1_omega = dt * \
                    (-(g/l*theta_values[i]) - 2*gamma *
                    omega_values[i]+alpha_d*np.sin(omega_d*t_values[i]))

                k2_theta = dt * (omega_values[i] + 0.5 * k1_omega)
                k2_omega = dt * (-(g / l) * theta_values[i] - 2 * gamma * omega_values[i] + alpha_d * np.sin(omega_d * t_values[i]) + 0.5 * (-g / l) * (
                    theta_values[i] + 0.5 * k1_theta) - 2 * gamma * (omega_values[i] + 0.5 * k1_omega) + alpha_d * np.sin(omega_d * (t_values[i] + 0.5 * dt)))

                k3_theta = dt * (omega_values[i] + 0.5 * k2_omega)
                k3_omega = dt * (-(g / l) * (theta_values[i] + 0.5 * k2_theta) - 2 * gamma * (
                    omega_values[i] + 0.5 * k2_omega) + alpha_d * np.sin(omega_d * (t_values[i] + 0.5 * dt)))

                k4_theta = dt * (omega_values[i] + k3_omega)
                k4_omega = dt * (-(g / l) * (theta_values[i] + k3_theta) - 2 * gamma * (
                    omega_values[i] + k3_omega) + alpha_d * np.sin(omega_d * (t_values[i] + dt)))

                # Update values
                theta_values.append((k1_theta + 2 * k2_theta + 2 *
                                    k3_theta + k4_theta) / 6.0 + theta_values[i])
                omega_values.append((k1_omega + 2 * k2_omega + 2 *
                                    k3_omega + k4_omega) / 6.0 + omega_values[i])
                t_values.append(t_values[i]+dt)

                # print(omega_values[i],theta_values[i])
                i = i+1

        return theta_values,omega_values,t_values


    def phase_shift(omega):
            right = False
            x_left = 0
            x_right = 0
            for i in range(1850,2000):
                if omega[i] < 0 and omega[i+2]>0:
                    x_left = i+1
                    right =True
                
                if right:
                    if omega[i] > 0 and omega[i+2]<0:
                        x_right = i+1

            return x_left, x_right


    for i in range(len(part_list)):
        if part_list[i] ==2:
            theta_1,omega_1,t = euler_cromer()
            theta_values,omega_values,t_values = rk4()

         
            plt.figure(1)
            plt.plot(t, theta_1, label="Euler-Cromer")
            plt.plot(t_values, theta_values, label="RK4")
            # plt.plot(t, theta_2, label="tehta2o = 3")
            plt.title("θ (t) for the Euler–Cromer and RK4")
            plt.legend()
            plt.xlabel('Time(sec)')
            plt.ylabel('Theta(rad)')
            plt.savefig("Oscilatory-Part2: θ (t) for the Euler–Cromer and RK4.pdf")

            plt.figure(2)
            plt.plot(t, omega_1, label="Euler-Cromer")
            plt.plot(t_values, omega_values, label="RK4")
            # plt.plot(t, omega_2, label="omega2")
            plt.title("ω (t) for the Euler–Cromer and RK4 methods")
            plt.legend()
            plt.xlabel('Time(sec)')
            plt.ylabel('ω(rad/s)')
            plt.savefig("Oscilatory-Part2: ω (t) for the Euler–Cromer and RK4 methods.pdf")


            driving_freq = np.linspace(0,2,10).tolist()
            amplitude = []
            
            for omega_d in driving_freq:
                theta_values,omega_values,t_values = rk4(0,0,omega_d)
            
                amplitude.append(max(omega_values[500:]))
                # if omega_d != 0:
                #     x,y = phase_shift (omega_values)
                #     print(x,y)
                #     print((y-1850)*dt)

            plt.figure(3)
            plt.plot(driving_freq, amplitude, marker='o', label="Amplitude From Plot Using RK4")
            plt.title("Amplitude Vs Driving Frequency ")
            plt.legend()
            plt.xlabel('Driving Frequency(rad/sec)')
            plt.ylabel('Amplitude(rad)')
            plt.savefig("Oscilatory-Part2: Amplitude Vs Driving Frequencys.pdf")
            plt.show()
        

        elif part_list[i] ==3:
            # Part 3
            theta0 = 0.1
            
            t =  []
            omega_1 =[]
            theta_1 = []

            t.append(0)
            theta_1.append(theta0)
            omega_1.append(omega0)

            KE = []
            PE = []
            TE = []
            KE.append(0)
            PE.append((0.5*g*l*(theta_1[0])**2))
            TE.append((KE[0]+PE[0]))

            i = 1

            while i < 1000:
                omega_1.append(-(g/l*(theta_1[i-1])*dt) - 2*gamma*omega_1[i-1]
                            * dt+alpha_d*np.sin(omega_d*t[i-1])*dt+omega_1[i-1])
                theta_1.append(theta_1[i-1]+omega_1[i]*dt)
                t.append(t[i-1]+dt)

                KE.append(0.5 * l**2 * (omega_1[i])**2)
                PE.append(0.5 * g * l * (theta_1[i])**2)
                # print(theta_1[i],omega_1[i], KE[i], PE[i])
                TE.append(PE[i] + KE[i])
                
                i = i+1

            plt.figure(4)
            plt.plot(t, PE, label="Potential Energy")
            plt.plot(t, KE, label="Kinetic Energy")
            plt.plot(t, TE, label="Total Energy")
            plt.title("Kinetic, potential and total energy when using the Euler–Cromer method")
            plt.xlabel('Time(sec)')
            plt.ylabel('Energy(J/Kg)')
            plt.legend()
            plt.savefig("Oscilatory-Part3: Kinetic, potential and total energy when using the Euler–Cromer method.pdf")

            plt.show()
        

        elif part_list[i] ==4:
            # Part 4
            alpha_d_list = [0.2, 1.2]
            t = []
            theta_1 = []
            omega_1 = []
            
            for alpha_d in alpha_d_list:
                theta_1,omega_1,t = euler_cromer(0,0,0.8, effect = "Non-Linear")
                theta_values,omega_values,t_values = rk4(0,0,0.8, effect = "Non-Linear")

                plt.figure(5)
                plt.plot(t, theta_1, label=f"Euler-Cromer alpha_d = {alpha_d}")
                plt.plot(t_values, theta_values, label=f"RK4 alpha_d = {alpha_d}")
                # plt.plot(t, theta_2, label="tehta2o = 3")
                plt.title("θ (t) for the Euler–Cromer and RK4 [Non Linear vs Linear]")
                plt.xlabel('Time(sec)')
                plt.ylabel('Theta(rad)')
                plt.legend()
                plt.savefig("Oscilatory-Part4: θ (t) for the Euler–Cromer and RK4 [Non Linear vs Linear].pdf")

                plt.figure(6)
                plt.plot(t, omega_1, label=f"Euler-Cromer alpha_d = {alpha_d}")
                plt.plot(t_values, omega_values, label=f"RK4 alpha_d = {alpha_d}")
                # plt.plot(t, omega_2, label="omega2")
                plt.title("ω (t) for the Euler–Cromer and RK4 methods [Non Linear vs Linear]")
                plt.xlabel('Time(sec)')
                plt.ylabel('ω (rad/sec)')
                plt.legend()
                plt.savefig("Oscilatory-Part4: ω (t) for the Euler–Cromer and RK4 methods [Non Linear vs Linear].pdf")

            plt.show()

        elif part_list[i] == 5:
            # Part 5

            omega_d_pt5 = 0.666
            delta_theta = np.linspace(0,0.005,5).tolist()
            alpha_d_list = [0.2,0.5,1.2]
            t_values = []
            theta_values = []
            omega_values = []
            theta_lst = []
            theta_1 = []
            theta_2 = []
            y = []
            
            for j in range(len(delta_theta)-1):
                for alpha_d in alpha_d_list:
                    del theta_lst [:]
                    del theta_1 [:]
                    del theta_2 [:]
                    del y [:]
                    theta_lst = [delta_theta[j],delta_theta[j+1]]

                    theta_1,omega_values,t_values = rk4(delta_theta[j],0,0.666, effect = "Non-Linear")
                    theta_2,omega_values,t_values = rk4(delta_theta[j+1],0,0.666, effect = "Non-Linear")

                    

                    for k in range(len(theta_1)):
                        lhs = abs((theta_1[k] - theta_2[k])/(theta_1[0] - theta_2[0]))
                        # print(lhs)

                        # lhs_log = math.log(lhs)
                        y.append(lhs)

                    plt.figure()
                    plt.semilogy(t_values , y,label =f"{theta_1[0]},{theta_2[0]},{alpha_d}")
                    slope, intercept = np.polyfit(t_values, y, 1)
                    print(slope,intercept)
                    plt.title(f"Plot for intial |{theta_1[0]} - {theta_2[0]}| with alpha_d(rad/sec) = {alpha_d} using RK4")
                    plt.xlabel('Time(sec)')
                    plt.ylabel('ω (rad/sec)')
                    plt.legend()
                    plt.savefig(f"Oscilatory-Part5: |{theta_1[0]} - {theta_2[0]}| with alpha_d (rad_sec) = {alpha_d} using RK4.pdf")
            
                    # plt.show()
    
if __name__ == "__main__":
    oscillator() # Calling the function (make oscillator PART=1,2,3)
