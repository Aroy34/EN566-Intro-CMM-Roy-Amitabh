import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import argparse as argp


psr = argp.ArgumentParser("Oscilatory")
psr.add_argument('--part', type=str, default=0,
                 help="enter the part ','")  # python Oscillator.py --part 1,2,3
arg = psr.parse_args()
part_str = arg.part.split(",")
part_list = [int(pt) for pt in part_str]  # list fo all the step widths

print(part_list)


def oscillator():
    """The function can do claculations using euler-cromer
    RK4 for an pendulum eiter having a liner or non-linear effect"""
    l = 9.8 # Length in meter
    g = 9.8 # Accelaration due to gravity
    gamma = 0.25 # 1/sec
    alpha_d = 0.2 # rad/sec^2
    theta0 = 0 # rad
    omega0 = 0 # rad/sec
    omega_d = 0.8 # closer to the ressonance frequency (rad/sec)
    dt = 0.05 # Time step size

    # Euler cromer
    def euler_cromer(theta0=0, omeg0=0, omegd=0.8, effect="Linear"):
        """The function can do claculations using Euler-Cromer for an 
        pendulum eiter having a liner or non-linear effect.
        
        theta0 : Initial guess for angle
        omeg0 : Initial guess for omega
        omegd: Value for omega_d
        effect : Linear or Non-Linear effect
        
        Returns: time, theta,omega        
        """
        theta0_1 = theta0
        omega0 = omeg0
        omega_d = omegd
        # if effect == "non-linear":
        t = [] # List to store time
        theta_1 = [] # List to store theta
        omega_1 = [] # List to store omega

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
                print(theta_1[i], theta_1[i+1])
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
        return theta_1, omega_1, t

    # RK4
    def rk4(theta0=0, omeg0=0, omegd=0.8, effect="Linear"):
        """The function can do claculations using RK4 for an 
        pendulum eiter having a liner or non-linear effect.
        
        theta0 : Initial guess for angle
        omeg0 : Initial guess for omega
        omegd: Value for omega_d
        effect : Linear or Non-Linear effect

        Returns: time, theta,omega
        
        """
        theta0_1 = theta0
        omega0 = omeg0
        omega_d = omegd

        t_values = [] # List to store time
        theta_values = [] # List to store theta
        omega_values = [] # List to store omega

        del t_values[:]
        del theta_values[:]
        del omega_values[:]

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

        return theta_values, omega_values, t_values

    def fit_sin(tt, yy):
        """
        Function take tiem and theta list to fit a sin curve 
        and then return aplitude and phase
        
        returns: fitting parameters "amp", "omega", "phase",
        "offset", "freq", "period" and "fitfunc
        """
        tt = np.array(tt) # Time array
        yy = np.array(yy) # Theta array
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        # excluding the zero frequency "peak", which is related to offset
        guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

        def sinfunc(t, A, w, p, c): return A * np.sin(w*t + p) + c
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        def fitfunc(t): return A * np.sin(w*t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc}

    for i in range(len(part_list)):
        if part_list[i] == 2:
            theta_1, omega_1, t = euler_cromer()
            theta_values, omega_values, t_values = rk4()

            plt.figure(1)
            plt.plot(t, theta_1, label="Euler-Cromer")
            plt.plot(t_values, theta_values, label="RK4")
            # plt.plot(t, theta_2, label="tehta2o = 3")
            plt.title("θ (t) for the Euler–Cromer and RK4")
            plt.legend()
            plt.xlabel('Time(sec)')
            plt.ylabel('Theta(rad)')
            plt.savefig(
                "Oscilatory-Part2: θ (t) for the Euler–Cromer and RK4.pdf")

            plt.figure(2)
            plt.plot(t, omega_1, label="Euler-Cromer")
            plt.plot(t_values, omega_values, label="RK4")
            # plt.plot(t, omega_2, label="omega2")
            plt.title("ω (t) for the Euler–Cromer and RK4 methods")
            plt.legend()
            plt.xlabel('Time(sec)')
            plt.ylabel('ω(rad/s)')
            plt.savefig(
                "Oscilatory-Part2: ω (t) for the Euler–Cromer and RK4 methods.pdf")

            driving_freq = np.linspace(0, 2, 10).tolist()
            amplitude = []
            phase =[]
            phase_shift = []

            for omega_d in driving_freq:
                theta_values, omega_values, t_values = rk4(0, 0, omega_d)

                res = fit_sin(t_values[1000:], theta_values[1000:])

                amplitude.append(abs(res['amp']))
                phase.append((res['phase']))

                # plt.figure()
                # plt.plot(t_values, theta_values, label=f'Omega_D = {omega_d}')
                # plt.plot(t_values[1000:], res['fitfunc'](np.array(t_values[1000:])), label='Fitted Sin Wave', linestyle='--')
                # plt.title(f"θ (t) using RK4 for omega_D = {omega_d}")
                # plt.legend()
                # plt.xlabel('Time(sec)')
                # plt.ylabel('Theta(rad)')
                # plt.savefig(
                #     f"Oscilatory-Part2: Fitted sin wave for omega_D = {omega_d}.pdf")
            
            for i in range(len(phase)-1):
                phase_shift.append((phase[0]-phase[i+1]))
            
            plt.figure(3)
            plt.plot(driving_freq, amplitude, marker='o',
                     label="Amplitude From Plot Using RK4")
            plt.title("Amplitude Vs Driving Frequency ")
            plt.legend()
            plt.xlabel('Driving Frequency(rad/sec)')
            plt.ylabel('Amplitude(rad)')
            plt.savefig(
                "Oscilatory-Part2: Amplitude Vs Driving Frequencys.pdf")
            
            plt.figure(4)
            plt.plot(driving_freq, phase, marker='o',
                     label="Phase From Plot Using RK4")
            plt.title("Phase Vs Driving Frequency ")
            plt.legend()
            plt.xlabel('Driving Frequency(rad/sec)')
            plt.ylabel('Phase (rad)')
            plt.savefig(
                "Oscilatory-Part2: Phase Vs Driving Frequencys.pdf")
            plt.show()

            

        elif part_list[i] == 3:
            # Part 3
            theta0 = 0.1 # Initial guess for the angle

            l = 9.8 # Length in meter
            g = 9.8 # Accelaration due to gravity
            t_en = []
            omega_1_en = []
            theta_1_en = []

            t_en.append(0)
            theta_1_en.append(theta0)
            omega_1_en.append(0)

            KE = []
            PE = []
            TE = []
            KE.append(0)
            PE.append((0.5*g*l*(theta_1_en[0])**2))
            TE.append((KE[0]+PE[0]))

            i = 1

            while i < 1000:
                omega_1_en.append(-(g/l*(theta_1_en[i-1])*dt) - 2*gamma*omega_1_en[i-1]
                               * dt+alpha_d*np.sin(omega_d*t_en[i-1])*dt+omega_1_en[i-1])
                theta_1_en.append(theta_1_en[i-1]+omega_1_en[i]*dt)
                t_en.append(t_en[i-1]+dt)

                KE.append(0.5 * l**2 * (omega_1_en[i])**2)
                PE.append(0.5 * g * l * (theta_1_en[i])**2)
                # print(theta_1[i],omega_1[i], KE[i], PE[i])
                TE.append(PE[i] + KE[i])

                i = i+1

            plt.figure(5)
            plt.plot(t_en, PE, label="Potential Energy")
            plt.plot(t_en, KE, label="Kinetic Energy")
            plt.plot(t_en, TE, label="Total Energy")
            plt.title(
                "Kinetic, potential and total energy when using the Euler–Cromer method")
            plt.xlabel('Time(sec)')
            plt.ylabel('Energy(J/Kg)')
            plt.legend()
            plt.savefig(
                "Oscilatory-Part3: Kinetic, potential and total energy when using the Euler–Cromer method.pdf")

            plt.show()

        elif part_list[i] == 4:
            # Part 4
            alpha_d_list = [0.2, 1.2] # List of alpha_d
            t = []
            theta_1 = []
            omega_1 = []

            for alpha_d in alpha_d_list:
                theta_1, omega_1, t = rk4(
                    0, 0, 0.8)
                theta_values, omega_values, t_values = rk4(
                    0, 0, 0.8, effect="Non-Linear")

                plt.figure(6)
                plt.plot(t, theta_1, label=f"Linear = {alpha_d}")
                plt.plot(t_values, theta_values,
                         label=f"Non-linear = {alpha_d}")
                # plt.plot(t, theta_2, label="tehta2o = 3")
                plt.title(
                    "θ (t) for the RK4 ")
                plt.xlabel('Time(sec)')
                plt.ylabel('Theta(rad)')
                plt.legend()
                plt.savefig(
                    "Oscilatory-Part4: θ (t) for the RK4 [Non Linear vs Linear].pdf")

                plt.figure(7)
                plt.plot(t, omega_1, label=f"Linear = {alpha_d}")
                plt.plot(t_values, omega_values,
                         label=f"Non-Linear = {alpha_d}")
                # plt.plot(t, omega_2, label="omega2")
                plt.title(
                    "ω (t) for the RK4 methods ")
                plt.xlabel('Time(sec)')
                plt.ylabel('ω (rad/sec)')
                plt.legend()
                plt.savefig(
                    "Oscilatory-Part4: ω (t) for the RK4 methods [Non Linear vs Linear].pdf")

            plt.show()

        elif part_list[i] == 5:
            # Part 5

            delta_theta = np.linspace(0, 0.003, 4).tolist() # List of theta values
            alpha_d_list = [0.2, 0.5, 1.2] # List of alpha_d
            t_values = [] # List to store time
            theta_values = [] # List to store theta
            omega_values = [] # List to store omega
            theta_lst = [] 
            theta_1 = []
            theta_2 = []
            y = [] # List to y-coordinate

            for j in range(len(delta_theta)-1):
                for alpha_d in alpha_d_list:
                    del theta_lst[:]
                    del theta_1[:]
                    del theta_2[:]
                    del y[:]
                    theta_lst = [delta_theta[0], delta_theta[j+1]]

                    theta_1, omega_values, t_values = rk4(
                        delta_theta[0], 0, 0.666, effect="Non-Linear")
                    theta_2, omega_values, t_values = rk4(
                        delta_theta[j+1], 0, 0.666, effect="Non-Linear")

                    for k in range(len(theta_1)):
                        lhs = abs((theta_1[k] - theta_2[k]) /
                                  (theta_1[0] - theta_2[0]))
                        # print(np.log(lhs))
                        y.append(np.log(lhs))


                    plt.figure()
                    plt.plot(
                        t_values, y)

                    slope,intercept = np.polyfit(t_values[200:700], y[200:700], 1)
                    z = np.polyfit(t_values[200:700], y[200:700], 1)
                    p = np.poly1d(z)
                    
                    
                    plt.plot(t_values[200:700],p(t_values[200:700]), label=f"λ = {slope} (1/sec)")
                    plt.title(
                        f"Plot for intial |{theta_1[0]} - {theta_2[0]}| with alpha_d(rad/sec) = {alpha_d} using RK4")
                    plt.xlabel('Time(sec)')
                    plt.ylabel('ω (rad/sec)')
                    plt.legend()
                    plt.savefig(
                        f"Oscilatory-Part5: |{theta_1[0]} - {theta_2[0]}| with alpha_d (rad_sec) = {alpha_d} using RK4.pdf")

                    plt.show()


if __name__ == "__main__":
    oscillator()  # Calling the function (make oscillator PART=1,2,3)
