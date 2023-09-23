"""
Name: CP_PS3_Q2.py
Author: Sandhya Sharma
Date: September 21, 2023
Description: Performing an NxN matrix multiplication two ways: using nested loops and using np.dot() and comparing their runtimes as N increases. 

"""

#importing necessary libraries 
import numpy as np 
import timeit
import matplotlib.pyplot as plt
import math

#function for matrix multiplication
def multiply(N, A,B):
    result = np.zeros([N,N] ,int)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i,j] += A[i,k]*B[k,j]
    return result

N = np.arange(10, 100, 10)

loop_time = np.empty(shape = (0,))
dot_time = np.empty(shape = (0,))

for i in range(N.size):
    #creating matrices A and B of random integers < 20 of size N x N
    A = np.random.randint(20, size = (N[i], N[i]))
    B = np.random.randint(20, size = (N[i], N[i]))

    #using nested loops for multiplication and measuring runtime
    start1 = timeit.default_timer()
    C = multiply(N[i],A,B)
    stop1 = timeit.default_timer()
    loop_time = np.append(loop_time, stop1-start1)

    #using np.dot() for multiplication and measuring runtime
    start2 = timeit.default_timer()
    C_without_loop = np.dot(A,B)
    stop2 = timeit.default_timer()
    dot_time = np.append(dot_time, stop2-start2)

#plotting runtimes against number of operations
fig, ((ax1, ax2) , (ax3, ax4)) = plt.subplots(2,2)
ax1.scatter(N, loop_time, color = 'black')
ax1.set(xlabel = 'N (no unit)', ylabel = ("Runtime(s)"))
ax1.set_title("Runtime v/s Size of Matrix for Loop")

ax2.scatter(N, dot_time, color = 'orange')
ax2.set(xlabel = 'N (no unit)', ylabel = ("Runtime(s)"))
ax2.set_title("Runtime v/s Size of Matrix for dot()")

#plotting log10 of runtimes against lo10 of number of operations to see if the complexity increases with N**3 
ax3.scatter(np.log10(N), np.log10(loop_time), color = 'black')
m1, b1 = np.polyfit(np.log10(N), np.log10(loop_time), deg=1) #finding the slope of the linear fit
ax3.set(xlabel = 'log10(N) (no unit)', ylabel = ("log10(runtime(s))"))
ax3.plot(np.log10(N), m1*np.log10(N) + b1, label = 'Slope =' + str(float(f'{m1:.2f}')))
ax3.legend(loc = "lower right")

ax4.scatter(np.log10(N), np.log10(dot_time), color = 'orange')
m2, b2 = np.polyfit(np.log10(N), np.log10(dot_time), deg=1) #finding the slope of the linear fit
ax4.set(xlabel = 'log10(N) (no unit)', ylabel = ("log10(runtime(s))"))
ax4.plot(np.log10(N), m2*np.log10(N) + b2, label = 'Slope =' + str(float(f'{m2:.2f}')))
ax4.legend(loc = "lower right")   
plt.show()








