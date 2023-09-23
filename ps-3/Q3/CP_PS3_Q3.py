"""
Name: CP_PS3_Q3.py
Author: Sandhya Sharma
Date: September 23, 2023
Description: Performing the radioactive decay of 10,000 Bi-213 atoms into Pb-209 and Tl-209 and subsequently Bi-209 for 20,000 seconds. 

"""

#importing necessary libraries 
import math 
import numpy as np 
import random
import matplotlib.pyplot as plt

#initializing variables required to simulate the radiaoactive decay
n_bi213 = 10000; n_pb = 0; n_tl = 0; n_bi209 = 0 #initial number of four isotopes
total_time = 20000 #total time of simulation
time_step = 1 #length of time for each step 
tau_bi213 = 46*60 #half-life of Bi-213
tau_pb = 3.3*60 #half-life of Pb-209
tau_tl = 2.2*60  #half-life of Tl-209
prob_bi213_to_pb = 0.9791 #probability that one atom of Bi-213 decays into Pb-209 at any given time
prob_bi213_to_pl = 0.209 #probability that one atom of Bi-213 decays into Tl-209 at any given time

#initializing arrays to record the number of each isotope over time
time_array = np.arange(0,total_time, time_step)
n_bi213_array = np.empty(shape = (0,))
n_pb_array = np.empty(shape = (0,))
n_tl_array = np.empty(shape = (0,))
n_bi209_array = np.empty(shape = (0,))

#function to return the probability of decay of any one atom (determined by the value of tau) in the span of one second
def probability_of_decay(tau):
    p = 1 - 2**(-1/tau)
    return p 

#performing every step of radioactive decay for four isotopes for each unit of tine
for i in range(total_time):
    n_bi213_array = np.append(n_bi213_array, n_bi213)
    n_pb_array = np.append(n_pb_array, n_pb)
    n_tl_array = np.append(n_tl_array, n_tl)
    n_bi209_array = np.append(n_bi209_array, n_bi209)

    for j in range(n_bi213):
        if random.random() < probability_of_decay(tau_bi213):
            n_bi213 -= 1
            if random.random() < prob_bi213_to_pb:
                n_pb += 1
            else:
                n_tl += 1

    for j in range(n_tl):
        if random.random() < probability_of_decay(tau_tl):
            n_tl -= 1
            n_pb += 1
            
    for j in range(n_pb):
        if random.random() < probability_of_decay(tau_pb):
            n_pb -= 1
            n_bi209 += 1

#recording the initial and final number of atoms for each isotope 
print('Initial Number of Atoms (at t = 0 s): ')
print('Bi-213 = ', n_bi213_array[0])
print('Pb-209 = ', n_pb_array[0])
print('Tl-209 = ', n_tl_array[0])
print('Bi-209 = ',  n_bi209_array[0])
print()

print('Final number of atoms (at t = 20000 s): ')
print('Bi-213 = ', n_bi213)
print('Pb-209 = ', n_pb)
print('Tl-209 = ', n_tl)
print('Bi-209 = ', n_bi209)
print()

#plotting the number of atoms of each isotope against time 
plt.plot(time_array, n_bi213_array, color = 'sandybrown', label = 'Bi-213')
plt.plot(time_array, n_bi209_array, color = 'thistle', label = 'Bi-209')
plt.plot(time_array, n_pb_array, color = 'olivedrab', label = 'Pb')
plt.plot(time_array, n_tl_array, color = 'lightsteelblue', label = 'Tl')
plt.legend(loc = 'upper right')
plt.title('Radioactive Decay of Bismuth-213 to Bismuth-209')
plt.xlabel('Time (s)')
plt.ylabel('Number of atoms of four isotopes (no units)')
plt.savefig('cp_ps3_q3.png')
plt.show()






                
