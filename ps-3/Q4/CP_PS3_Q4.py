"""
Name: CP_PS3_Q4.py
Author: Sandhya Sharma
Date: September 23, 2023
Description: Performing the radioactive decay of 1000 atoms of Tl-208 to Pb-208 using non-uniform distribution. 

"""

#importing necessary libraries 
import math 
import numpy as np 
import random
import matplotlib.pyplot as plt

#initializing values required for radioactive decay
n_tl208 = 1000
tau_tl209 = 3.053*60
mu = math.log(2)/tau_tl209
total_time = 1000
time_step = 1
time_array = np.arange(0,total_time, time_step)

#initializing an array to store the random numbers generated in the next loop
t_array = np.empty(shape = (0,))

#generating random numbers from a non-uniform distribution that represents the time at which a given atom will decay 
for i in range(n_tl208):
    t = -(1/mu)*math.log(1-random.random())
    t_array = np.append(t_array, t)

#sorting the array and finding the number of atoms that decay before a given time
t_array_sorted = np.sort(t_array)
decayed_atoms = 0 
n_tl208_array = np.empty(shape = (0,))

for i in range(total_time):
    decayed_atoms = np.argmax(t_array_sorted > i)
    remaining_n_tl208 = n_tl208 - decayed_atoms
    n_tl208_array = np.append(n_tl208_array, remaining_n_tl208)

#plotting the number of atoms of Tl-208 against time 
plt.plot(time_array, n_tl208_array, color = 'maroon')
plt.title('Radioactive Decay of Thallium-208 to Lead-208')
plt.xlabel('Time(s)')
plt.ylabel('Number of Tl-208 atoms (no units)')
plt.savefig('cp_ps3_q4')
plt.show()
        
    

