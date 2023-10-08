# Import matplotlib.pyplot for plotting the graph
# import numpy for the interpolation
# Import math for doing mathermatical operations
# Import argparse for passing arguments from the terminal to python code
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse as argp


def beta_decay():
    """ This functions take argument --plot=step_width.The code calculate
    the rate of radioactive decay  plot it for the different step_width.
    Code also interpolates the value of rate at 2 times of half life.
    Code will also caluclate the condition when the 2nd order of taylor 
    series is taken into consideration for the larger step width """
    psr = argp.ArgumentParser("Activity")
    psr.add_argument('--plot', type=str, default=0,
                     help="enter the step widths ','")
    arg = psr.parse_args()
    step_ = arg.plot.split(",")
    step_size = [int(stp) for stp in step_]  # list fo all the step widths
    # Initial number of particles
    initial_particles = (10**-12/14)*6.022*10**23
    # t(1/2) half life in years
    half_life = 5700
    # 2* t(1/2) half life in years
    half_life_2 = 5700*2
    # Decay constant
    tau = -half_life/math.log(0.5)
    time = []  # List for storing time
    number_particle_at_t = []  # List for number of particles
    rate = []  # List for storing rate
    '''For loop for cyclying though each step width'''
    for i in range(len(step_size)):
        # Emptying the list
        del time[:]
        del number_particle_at_t[:]
        del rate[:]
        '''For loop for cyclying though each time stamp'''
        for t in range(0, 20000, step_size[i]):
            if t == 0:
                number_particle_at_t.append(
                    initial_particles)  # Adding the t=o value
                time.append(t)  # Adding the t=o value
            else:
                # Calculating the index value for the list
                index = int((t)/step_size[i])
                number_particle_at_t.append(number_particle_at_t[index-1]-(
                    1/tau*number_particle_at_t[index-1] *
                    step_size[i]))  # adding the next value
                # adding the next value
                rate.append(-(number_particle_at_t[index] -
                            number_particle_at_t[index-1])/step_size[i])
                time.append(t)  # adding the next value
            if step_size[i] >= 1000:  # checking for a special condition of step width is 1000
                # List for storing number of particle when 2nd order of taylor expansion is considered
                no_2nd_order = []
                time_2nd_order = []  # List for storing time
                rate_2nd_order = []  # List for storing rate
                for t in range(0, 20000, step_size[i]):
                    if t == 0:
                        no_2nd_order.append(initial_particles)
                        time_2nd_order.append(t)
                    else:
                        index = int((t)/step_size[i])
                        no_2nd_order.append(no_2nd_order[index-1] -
                                            (1/tau*no_2nd_order[index-1]*step_size[i]) +
                                            (1/tau**2*no_2nd_order[index-1]*step_size[i]**2)/2)  # Considering the 2nd order term of the taylor expansion
                        rate_2nd_order.append(
                            -(no_2nd_order[index]-no_2nd_order[index-1])/step_size[i])
                        time_2nd_order.append(t)

        plt.plot(time[:-1], rate, label=" Step_Width = " +
                 step_[i])  # Plotting normal condition
        if step_size[i] == 1000:  # Plotting special condition
            plt.plot(time_2nd_order[:-1], rate_2nd_order,
                     label=" Step_Width = "+step_[i] + " 2nd order considered")
        # interpolating the value of rate
        rate_interp = round(np.interp(half_life_2, time[:-1], rate), 2)
        print("Rate at " + str(half_life_2) + " years = " +
              str(rate_interp) + " ;for step width = " + str(step_size[i]))
        if step_size[i] == 1000:
            # interpolating the value of rate
            rate_interp = round(
                np.interp(half_life_2, time_2nd_order[:-1], rate_2nd_order), 2)
            print("Rate at " + str(half_life_2) + " years = " + str(rate_interp) +
                  " ;for step width = " + str(step_size[i])+" considering the 2nd order")

    plt.xlabel("Time (Years)")
    plt.ylabel("Rate * 10^6 (Number of particles/years)")
    plt.legend()
    plt.title("Rate of Radioactive Decay,")
    plt.savefig("Carbon_plot.jpeg")
    plt.show()


if __name__ == "__main__":
    beta_decay()  # Calling the function (make carbon WIDTH=10,100,1000)
