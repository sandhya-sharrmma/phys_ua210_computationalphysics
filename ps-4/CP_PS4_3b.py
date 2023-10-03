"""
Name: CP_PS4_Q3b.py
Author: Sandhya Sharma
Date: September 30, 2023
Description: Finding the wavefunction of energy level n = 30 of the one-dimensional quantum harmonic oscillator. 

"""
#importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np 
import math

#function to return factorial of n without recursion
def factorial(n):
    if n == 0 or n == 1:
        result = 1
    else:
        result = 1
        for i in range(1,n+1):
            result *= i
    return result

#function to return nth Hermite polynomial without recursion
def H(n,x):
    if n == 0:
        return 1
    elif n == 2:
        return 2*x
    else:
        for i in range (2, n+1):
            h_0 = 1; h_1 = 2*x
            h_2 = 2*x*h_1 - 2*(i-1)*h_0
            h_0 = h_1
            h_1 = h_2
        return h_1

#function to return the wavefunction for given n
def psi(n,x):
    return (1/math.sqrt((2**n)*factorial(n)*math.sqrt(math.pi))) * math.exp(-(x**2)/2) * H(n,x)

#calculating wavefunction values for x ranging from -10 and 10 and n = 30
x_range = np.arange(-10,10, 0.01)
psi_30_range = np.empty(shape = (0, ))

for i in range(x_range.size):
    psi_30_range = np.append(psi_30_range, psi(30, x_range[i]))

#plotting wavefunction for given range of x and n = 30
plt.plot(x_range, psi_30_range, label = "n = 30")
plt.legend()
plt.title('Wavefunction of 30th Enery Level of a One-Dimensional Quantum Harmonic Oscillator')
plt.xlabel('x(m)')
plt.ylabel('$\Psi(n,x) (m^{-1/2})$')
plt.show()






