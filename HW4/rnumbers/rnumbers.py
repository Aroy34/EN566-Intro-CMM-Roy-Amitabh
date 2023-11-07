import random as random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def box_muller(rand_list,total_num,bin_size):
       
        u1 = rand_list[0:int(total_num/2)]
        u2 = rand_list[int(total_num/2):]
        # print(u1)
        
        z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

        # Combine both Z0 and Z1 to have the complete set of samples
        z = np.concatenate((z0, z1))
        
        # normally distributed random variable = mean *
        
        for bin in bin_size:
            plt.hist(z, bins=bin,density=True)
            xmin, xmax = plt.xlim() 
            x = np.linspace(xmin, xmax, 500) 
            p = norm.pdf(x, 0, 1) 
            plt.plot(x, p, 'r', linewidth=2,label = "Overlayed Gaussian") 
            plt.legend()
            # title = "Fit Values: {:.2f} and {:.2f}".format(mu, std) 
            plt.title(f"Gaussian distributed random numbers - {total_num} Samples, {bin} subdivisions")
            plt.savefig(f"Gaussian distributed random numbers - {total_num} Samples, {bin} subdivisions")
            
            plt.close()
            
            # plt.show()

def rnumbers(nos,bin_size):    
    
    for num_rand in nos:
        n =0
        num =[]
        r_num =[]
        num = np.random.rand(num_rand)
        print(num)
        
        box_muller(num,num_rand,bin_size)
        
        for bin in bin_size:
            plt.hist(num, bins=bin,density=True)
            plt.title(f"Random Numbers Distribution - {num_rand} Samples, {bin} subdivisions")
            plt.savefig(f"Random Numbers Distribution - {num_rand} Samples, {bin} subdivisions")
            plt.close()
            # plt.show()
        
if __name__ == "__main__":
    nos = [1000,1000000]
    # nos = [1000]
    bin_size = [10,20,50,100]
    rnumbers(nos,bin_size)
