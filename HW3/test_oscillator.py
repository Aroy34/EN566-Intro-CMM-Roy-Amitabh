import matplotlib.pyplot as plt
import math
import numpy as np

def oscillator():
    L = 9.8
    g = 9.8
    gamma = 0.25
    q = [2*gamma]
    alpha_d = 0.2
    theta0_1 = 0.11
    omega0 = 1
    omega_d = [0,1,2,3,4,5,6,7,8,9,10]
    dt = [0.001]
    t = []
    theta_1 = []
    omega_1 = []
    t.append(0)
    theta_1.append(theta0_1)
    omega_1.append(omega0)
    amplitudes = []
    phase_shifts = []

    for l in range(len(omega_d)):
        for k in range(len(q)):
            for j in range(len(dt)):
                i = 0
                del theta_1[:]
                del omega_1[:]
                del t[:]
                t.append(0)
                theta_1.append(theta0_1)
                omega_1.append(omega0)

                #Euler cromer
                while t[i] < 100:
                    omega_1.append(omega_1[i]-((g/L)*theta_1[i]*dt[j])-q[k]*omega_1[i]*dt[j]+alpha_d*np.sin((omega_d[l]*t[i]))*dt[j])
                    theta_1.append(theta_1[i]+omega_1[i+1]*dt[j])
                    t.append(t[i]+dt[j])
                    i =i+1
                
                plt.figure(1)
                plt.plot(t, theta_1, label=f"Driving force (Omega_d) = {omega_d[l]}")
                plt.title("When we use Euler-cramer, with a driving force ")
                plt.legend()
        phase_= []
        amplitude = max(theta_1)
        peak_index = np.argmax(theta_1)
        phase_.append(t[peak_index])
        for i in range(1,len(phase_)):
            phase_shifts.append(phase_[i]-phase_[0])


        
    
    plt.figure(2)
    plt.plot(omega_d, amplitudes, label= "amplitude")
    plt.figure(3)
    plt.plot(omega_d[1:], phase_shifts, label= "phase s")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    oscillator()