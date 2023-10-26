import random
import numpy as np
import matplotlib.pyplot as plt
import math

# print(np.concatenate([5, -1]).cumsum(0))


def walk(step,origin_x,origin_y, show):
    x = origin_x
    y = origin_y
    position = []
    if show:
        x_ls = [x]
        y_ls = [y]
    for i in range(step):
        rand = random.uniform(0,1)
        if rand <= 0.25:
            dx, dy = 0.1, 0
        elif rand <= 0.5:
            dx, dy = - 0.1, 0
        elif rand <= 0.75:
            dx, dy = 0, 0.1
        else:
            dx, dy = 0, -0.1

        x = x+dx 
        y = y+dy
        
        position.append(math.sqrt(x**2+y**2))

        if show:
            x_ls.append(x)
            y_ls.append(y)
    
    if show :
        plt.figure(3)
        plt.plot(x_ls,y_ls)
        plt.scatter(x_ls[0],y_ls[0], color = 'g')
        plt.scatter(x_ls[-1],y_ls[-1], color = 'r')
        



    return position, x, y


if __name__ == "__main__":
    step = 100
    walks_rand = 10000
    x,y = 0 ,0

    step_sw = 3
    pos_0_3, x,y=walk(step_sw,x,y,False)

    # print(pos,x,y)

    super_lst = [ ]

    plot = False
    for i in range(walks_rand):
        pos, _, _ = walk((step-step_sw), x, y,plot)
        plot = False
        super_lst.append(pos)

    

    lists_2d = np.array(super_lst)
    averages = np.mean(lists_2d, axis=0)
    x_avg = pos_0_3 + (averages.tolist())
    # print(x)
    x_sq_avg = []
    for i in range(len(x_avg)):
        x_sq_avg.append(x_avg[i]**2)
    


    n = (np.arange(0,step,1)).tolist()
    # print(n)
    plt.figure(1)
    plt.plot(n,x_avg)

    plt.figure(2)
    slope, intercept = np.polyfit(n, x_sq_avg, 1)
    print(type(slope))
    plt.scatter(n,x_sq_avg,s =1,  label = f"Slope = {round(slope,4)}")
    plt.legend()
    plt.show()
