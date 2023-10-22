import matplotlib.pyplot as plt
import numpy as np


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

def oscillator():
    l = 9.8
    g = 9.8
    gamma = 0.25
    alpha_d = 0.2
    theta0_1 = 0
    # theta0_2 = 0
    omega0 = 0
    omega_d = 0.8
    dt = 0.05


    # Euler cromer
    t = []
    theta_1 = []
    theta_2 = []
    omega_1 = []
    # omega_2 = []
    delta_t = []
    t.append(0)
    theta_1.append(theta0_1)
    # theta_2.append(theta0_2)
    omega_1.append(omega0)
    # omega_2.append(omega0)

    i = 0

    while t[i] < 100:
        omega_1.append(-(g/l*(theta_1[i])*dt) - 2*gamma*omega_1[i]
                       * dt+alpha_d*np.sin(omega_d*t[i])*dt+omega_1[i])
        # print(omega_1[i],omega_1[i+1])
        theta_1.append(theta_1[i]+omega_1[i+1]*dt)
        print(theta_1[i],theta_1[i+1])
        t.append(t[i]+dt)
        # print(omega_1[i+1],theta_1[i+1], t[i+1])

        # theta2
        # omega_2.append(-(g/l*(theta_2[i])*dt) - 2*gamma*omega_2[i]
        #                * dt+alpha_d*np.sin(omega_d*t[i])*dt+omega_2[i])
        # theta_2.append(theta_2[i]+omega_2[i+1]*dt)

        i = i+1

        # delta_t.append(abs(theta_1[i]-theta_2[i]))

        # if omega[i-1] == omega[i]:
        #     False
    # plt.figure(1)
    # plt.semilogx(t[1:], delta_t)
    # plt.title("Delta(theta1 - theta2)")
    # plt.legend()
    # plt.savefig("Delta(theta1 - theta2.pdf")

    # RK4
    t_values = []
    theta_values = []
    omega_values = []

    del t_values [:]
    del theta_values [:]
    del omega_values [:]

    t_values.append(0)
    theta_values.append(theta0_1)
    omega_values.append(omega0)

    i = 0

    while t_values[i] < 100:
        # Calculate the four RK4 increments
        k1_theta = dt * omega_values[i]
        k1_omega = dt * \
            (-(g/l*theta_values[i]) - 2*gamma *
            omega_values[i]+alpha_d*np.sin(omega_d*t[i]))

        k2_theta = dt * (omega_values[i] + 0.5 * k1_omega)
        k2_omega = dt * (-(g / l) * theta_values[i] - 2 * gamma * omega_values[i] + alpha_d * np.sin(omega_d * t[i]) + 0.5 * (-g / l) * (
            theta_values[i] + 0.5 * k1_theta) - 2 * gamma * (omega_values[i] + 0.5 * k1_omega) + alpha_d * np.sin(omega_d * (t[i] + 0.5 * dt)))

        k3_theta = dt * (omega_values[i] + 0.5 * k2_omega)
        k3_omega = dt * (-(g / l) * (theta_values[i] + 0.5 * k2_theta) - 2 * gamma * (
            omega_values[i] + 0.5 * k2_omega) + alpha_d * np.sin(omega_d * (t[i] + 0.5 * dt)))

        k4_theta = dt * (omega_values[i] + k3_omega)
        k4_omega = dt * (-(g / l) * (theta_values[i] + k3_theta) - 2 * gamma * (
            omega_values[i] + k3_omega) + alpha_d * np.sin(omega_d * (t[i] + dt)))

        # Update values
        theta_values.append((k1_theta + 2 * k2_theta + 2 *
                            k3_theta + k4_theta) / 6.0 + theta_values[i])
        omega_values.append((k1_omega + 2 * k2_omega + 2 *
                            k3_omega + k4_omega) / 6.0 + omega_values[i])
        t_values.append(t_values[i]+dt)

        # print(omega_values[i],theta_values[i])
        i = i+1

    driving_freq = np.linspace(0,2,10).tolist()
    amplitude = []
    
    for omega_d in driving_freq:
    
        del t_values [:]
        del theta_values [:]
        del omega_values [:]

        t_values.append(0)
        theta_values.append(theta0_1)
        omega_values.append(omega0)

        i = 0

        while t_values[i] < 100:
            # Calculate the four RK4 increments
            k1_theta = dt * omega_values[i]
            k1_omega = dt * \
                (-(g/l*theta_values[i]) - 2*gamma *
                omega_values[i]+alpha_d*np.sin(omega_d*t[i]))

            k2_theta = dt * (omega_values[i] + 0.5 * k1_omega)
            k2_omega = dt * (-(g / l) * theta_values[i] - 2 * gamma * omega_values[i] + alpha_d * np.sin(omega_d * t[i]) + 0.5 * (-g / l) * (
                theta_values[i] + 0.5 * k1_theta) - 2 * gamma * (omega_values[i] + 0.5 * k1_omega) + alpha_d * np.sin(omega_d * (t[i] + 0.5 * dt)))

            k3_theta = dt * (omega_values[i] + 0.5 * k2_omega)
            k3_omega = dt * (-(g / l) * (theta_values[i] + 0.5 * k2_theta) - 2 * gamma * (
                omega_values[i] + 0.5 * k2_omega) + alpha_d * np.sin(omega_d * (t[i] + 0.5 * dt)))

            k4_theta = dt * (omega_values[i] + k3_omega)
            k4_omega = dt * (-(g / l) * (theta_values[i] + k3_theta) - 2 * gamma * (
                omega_values[i] + k3_omega) + alpha_d * np.sin(omega_d * (t[i] + dt)))

            # Update values
            theta_values.append((k1_theta + 2 * k2_theta + 2 *
                                k3_theta + k4_theta) / 6.0 + theta_values[i])
            omega_values.append((k1_omega + 2 * k2_omega + 2 *
                                k3_omega + k4_omega) / 6.0 + omega_values[i])
            t_values.append(t_values[i]+dt)

            # print(omega_values[i],theta_values[i])
            i = i+1
        
        if False:
            plt.figure(5)
            plt.plot(t_values[500:1500], omega_values[500:1500],label= omega_d)
            plt.title("θ(Driving Force) resonance curve")
            plt.legend()

        amplitude.append(max(omega_values[500:]))
        
        if omega_d != 0:
            x,y = phase_shift (omega_values)
            print(x,y)
            print((y-1850)*dt)



    plt.figure(2)
    plt.plot(t, theta_1, label="Euler-Cromer")
    plt.plot(t_values, theta_values, label="RK4")
    # plt.plot(t, theta_2, label="tehta2o = 3")
    plt.title("θ (t) for the Euler–Cromer and RK4")
    plt.legend()
    plt.savefig("θ (t) for the Euler–Cromer and RK4.pdf")

    plt.figure(3)
    plt.plot(t, omega_1, label="Euler-Cromer")
    plt.plot(t_values, omega_values, label="RK4")
    # plt.plot(t, omega_2, label="omega2")
    plt.title("ω (t) for the Euler–Cromer and RK4 methods")
    plt.legend()
    plt.savefig("ω (t) for the Euler–Cromer and RK4 methods.pdf")

    plt.figure(4)
    plt.plot(driving_freq, amplitude, marker='o', label="fromt the plot")
    plt.title("θ(Driving Force) resonance curve")
    plt.legend()
    plt.show()



    # Part 2
    l = 9.8
    g = 9.8
    gamma = 0.25
    alpha_d = 0.2
    theta0_1 = 0.1
    omega0 = 0
    omega_d = 0.8
    dt = 0.05

    del t [:]
    del omega_1 [:]
    del theta_1[:]

    t.append(0)
    theta_1.append(theta0_1)
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


    plt.figure(6)
    plt.plot(t, PE, label="PE")
    plt.plot(t, KE, label="KE")
    plt.plot(t, TE, label="TE")
    plt.title("Kinetic, potential and total energy when using the Euler–Cromer method.")
    plt.legend()

    plt.show()

    # Part 3

    alpha_d_list = [0.2, 1.2]
    t = []
    theta_1 = []
    omega_1 = []
    
    for alpha_d in alpha_d_list:
        del t [:]
        del theta_1 [:]
        del omega_1 [:]
        
        t.append(0)
        theta_1.append(theta0_1)
        omega_1.append(omega0)

        
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

        # RK4
        t_values = []
        theta_values = []
        omega_values = []

        del t_values [:]
        del theta_values [:]
        del omega_values [:]

        t_values.append(0)
        theta_values.append(theta0_1)
        omega_values.append(omega0)

        i = 0

        while t_values[i] < 100:
            # Calculate the four RK4 increments
            k1_theta = dt * omega_values[i]
            k1_omega = dt * \
                (-(g/l*np.sin(theta_values[i])) - 2*gamma *
                omega_values[i]+alpha_d*np.sin(omega_d*t[i]))

            k2_theta = dt * (omega_values[i] + 0.5 * k1_omega)
            k2_omega = dt * (-(g / l) * np.sin(theta_values[i]) - 2 * gamma * omega_values[i] + alpha_d * np.sin(omega_d * t[i]) + 0.5 * (-g / l) * (
                np.sin(theta_values[i]) + 0.5 * k1_theta) - 2 * gamma * (omega_values[i] + 0.5 * k1_omega) + alpha_d * np.sin(omega_d * (t[i] + 0.5 * dt)))

            k3_theta = dt * (omega_values[i] + 0.5 * k2_omega)
            k3_omega = dt * (-(g / l) * (np.sin(theta_values[i]) + 0.5 * k2_theta) - 2 * gamma * (
                omega_values[i] + 0.5 * k2_omega) + alpha_d * np.sin(omega_d * (t[i] + 0.5 * dt)))

            k4_theta = dt * (omega_values[i] + k3_omega)
            k4_omega = dt * (-(g / l) * (np.sin(theta_values[i]) + k3_theta) - 2 * gamma * (
                omega_values[i] + k3_omega) + alpha_d * np.sin(omega_d * (t[i] + dt)))

            # Update values
            theta_values.append((k1_theta + 2 * k2_theta + 2 *
                                k3_theta + k4_theta) / 6.0 + theta_values[i])
            omega_values.append((k1_omega + 2 * k2_omega + 2 *
                                k3_omega + k4_omega) / 6.0 + omega_values[i])
            t_values.append(t_values[i]+dt)

            # print(omega_values[i],theta_values[i])
            i = i+1


        plt.figure(7)
        plt.plot(t, theta_1, label=f"Euler-Cromer alpha_d = {alpha_d}")
        plt.plot(t_values, theta_values, label=f"RK4 alpha_d = {alpha_d}")
        # plt.plot(t, theta_2, label="tehta2o = 3")
        plt.title("θ (t) for the Euler–Cromer and RK4 [Non Linear vs Linear]")
        plt.legend()
        plt.savefig("θ (t) for the Euler–Cromer and RK4[Non Linear vs Linear].pdf")

        plt.figure(8)
        plt.plot(t, omega_1, label=f"Euler-Cromer alpha_d = {alpha_d}")
        plt.plot(t_values, omega_values, label=f"RK4 alpha_d = {alpha_d}")
        # plt.plot(t, omega_2, label="omega2")
        plt.title("ω (t) for the Euler–Cromer and RK4 methods [Non Linear vs Linear]")
        plt.legend()
        plt.savefig("ω (t) for the Euler–Cromer and RK4 methods [Non Linear vs Linear].pdf")

    plt.show()



if __name__ == "__main__":
    oscillator()
