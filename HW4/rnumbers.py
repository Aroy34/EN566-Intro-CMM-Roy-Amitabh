import random as random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import argparse as argp


def box_muller(rand_list,total_num,bin_size):
       
        u1 = rand_list[0:int(total_num/2)]
        u2 = rand_list[int(total_num/2):]
        # print(u1)
        
        z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

        # Combine both Z0 and Z1 to have the complete set of samples
        z = np.concatenate((z0, z1))
        
        # normally distributed random variable = mean *
        
        def gaussian_pdf(x, mean=0, std_dev=1):
            return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-(x -mean)**2 / (2 *std_dev* 1**2))

    
        for bin in bin_size:
            plt.hist(z, bins=bin,density=True)
            xmin, xmax = plt.xlim() 
            x = np.linspace(xmin, xmax, 500) 
            p = gaussian_pdf(x)
            plt.plot(x, p, 'r', linewidth=2,label = "Overlayed Gaussian") 
            plt.legend()
            # title = "Fit Values: {:.2f} and {:.2f}".format(mu, std) 
            plt.title(f"Gaussian distributed random numbers - {total_num} Samples, {bin} subdivisions")
            plt.savefig(f"Gaussian distributed random numbers - {total_num} Samples, {bin} subdivisions")
            
            plt.close()
            
            plt.show()

def rnumbers():
    
    psr = argp.ArgumentParser("rnumbers")
    psr.add_argument('--part', type=str, default="1,2",
                     help="enter the part , ")  
    arg = psr.parse_args()
    part_str = arg.part.split(",")
    part_list = [int(pt) for pt in part_str]  # list fo all the step widths 
    nos = [1000,1000000]
    bin_size = [10,20,50,100]
    
    for num_rand in nos:
        n =0
        num =[]
        r_num =[]
        num = np.random.rand(num_rand)
        
        for part in part_list:
            if part == 1:
                for bin in bin_size:
                    plt.hist(num, bins=bin,density=True)
                    plt.title(f"Random Numbers Distribution - {num_rand} Samples, {bin} subdivisions")
                    plt.savefig(f"Random Numbers Distribution - {num_rand} Samples, {bin} subdivisions")
                    plt.close()
                    # plt.show()
            elif part == 2:
                box_muller(num,num_rand,bin_size)
    plt.show()
        
if __name__ == "__main__":
    rnumbers() # make rnumbers or make rnumbers PART=1,2
