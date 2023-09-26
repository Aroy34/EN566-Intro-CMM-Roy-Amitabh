import numpy as np
import matplotlib.pyplot as plt
import math


# Initial number of particles
initial_particles = (10**-12/14)*6.022*10**23
# t(1/2) half life in years
half_life = 5700
# Decay constant
tau = -half_life/math.log(0.5)
print(tau)

# Defining the memory allocation
dt = 10
dt2=100
dt3=1000

time = []
time2 = []
time3 = []
number_particle_at_t =[]
number_particle_at_t.append(10**-12/14*6.022*10**23)

number_particle =[]
number_particle.append(10**-12/14*6.022*10**23),

number =[]
number.append(10**-12/14*6.022*10**23),


for t in range(0,20000-dt,dt):
    index = int((t)/dt)
    number_particle_at_t.append(number_particle_at_t[index]-(1/tau*number_particle_at_t[index]*dt))

#print(len(number_particle_at_t))
for t in range(0,20000-dt2,dt2):
    index = int((t)/dt2)
    number_particle.append(number_particle[index]-(1/tau*number_particle[index]*dt2))

for t in range(0,20000-dt3,dt3):
    index = int((t)/dt3)
    number.append(number[index]-(1/tau*number[index]*dt3))


for y in range(0,20000,dt):
    time.append(y)

for y2 in range(0,20000,dt2):
    time2.append(y2)

for y3 in range(0,20000,dt3):
    time3.append(y3)

plt.plot(time, number_particle_at_t)
plt.plot(time2, number_particle)
plt.plot(time3, number)
plt.show()








