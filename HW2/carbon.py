import numpy as np
import matplotlib.pyplot as plt
import math
import argparse as argp

def beta_decay():
    psr = argp.ArgumentParser("Activity")
    psr.add_argument('--plot', type=str, default=0,
                     help="enter the step widths ','")
    arg = psr.parse_args()
    step_ = arg.plot.split(",")
    step_size = [int(stp) for stp in step_]
    # Initial number of particles
    initial_particles = (10**-12/14)*6.022*10**23
    # t(1/2) half life in years
    half_life = 5700
    # Decay constant
    tau = -half_life/math.log(0.5)
    print(tau)
    time = []
    number_particle_at_t =[]
    
    for i in range(len(step_size)):
        del time[:]
        del number_particle_at_t[:]
        number_particle_at_t.append(initial_particles)
        time.append(0)

        for t in range(0,20000+step_size[i],step_size[i]):
            if t == 0:
                number_particle_at_t.append(initial_particles)
            else:
                index = int((t)/step_size[i])
                number_particle_at_t.append(number_particle_at_t[index]-(1/tau*number_particle_at_t[index]*step_size[i]))

        for y in range(0,20000+step_size[i],step_size[i]):
            if y == 0:
                time.append(y)
            else:
                time.append(y)

        plt.plot(time, number_particle_at_t, label=" Step_Width = "+step_[i])

    plt.xlabel("Time (Years)")
    plt.ylabel("No of Particles (N * 10^5)")
    plt.legend()
    plt.title("Acivity of _____")
    plt.show()


if __name__ == "__main__":
    beta_decay()





