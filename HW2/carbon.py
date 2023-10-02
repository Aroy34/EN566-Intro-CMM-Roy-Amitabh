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
    half_life_2 = 5700*2
    tau = -half_life/math.log(0.5)
    time = []
    number_particle_at_t =[]
    rate = []
    
    for i in range(len(step_size)):
        del time[:]
        del number_particle_at_t[:]
        del rate[:]

        for t in range(0,20000,step_size[i]):
            if t == 0:
                number_particle_at_t.append(initial_particles)
                time.append(t)
            else:
                index = int((t)/step_size[i])
                number_particle_at_t.append(number_particle_at_t[index-1]-(1/tau*number_particle_at_t[index-1]*step_size[i]))
                #print(number_particle_at_t[index],number_particle_at_t[index-1])
                rate.append(-(number_particle_at_t[index]-number_particle_at_t[index-1])/step_size[i])
                time.append(t)
            if step_size[i] == 1000: 
                no_2nd_order = []
                time_2nd_order =[]
                rate_2nd_order = []
                for t in range(0,20000,step_size[i]):
                    if t == 0:
                        no_2nd_order.append(initial_particles)
                        time_2nd_order.append(t)
                    else:
                        index = int((t)/step_size[i])
                        no_2nd_order.append(no_2nd_order[index-1]-(1/tau*no_2nd_order[index-1]*step_size[i])+(1/tau**2*no_2nd_order[index-1]*step_size[i]**2)/2)
                        #print(number_particle_at_t[index],number_particle_at_t[index-1])
                        rate_2nd_order.append(-(no_2nd_order[index]-no_2nd_order[index-1])/step_size[i])
                        time_2nd_order.append(t)
                    


        plt.plot(time[:-1], rate, label=" Step_Width = "+step_[i])
        if step_size[i] == 1000: 
            plt.plot(time_2nd_order[:-1], rate_2nd_order, label=" Step_Width = "+step_[i]+ " 2nd order considered")
        rate_interp = round(np.interp(half_life_2, time[:-1], rate),2) 
        print("Rate at "+ str(half_life_2) +" years = "+ str(rate_interp) + " ;for step width = "+ str(step_size[i]))
        if step_size[i] == 1000: 
            rate_interp = round(np.interp(half_life_2, time_2nd_order[:-1], rate_2nd_order),2)
            print("Rate at "+ str(half_life_2) +" years = "+ str(rate_interp) + " ;for step width = "+ str(step_size[i])+" considering the 2nd order") 

       

    # print(number_particle_at_t)
    plt.xlabel("Time (Years)")
    plt.ylabel("Rate * 10^6 (Number of particles/years)")
    plt.legend()
    plt.title("Rate of Radioactive Decay,")
    plt.savefig("Carbon_plot.jpeg")
    plt.show()


if __name__ == "__main__":
    beta_decay()





