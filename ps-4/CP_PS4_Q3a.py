"""
Name: CP_PS4_Q3a.py
Author: Sandhya Sharma
Date: September 30, 2023
Description: Finding the wavefunction of nth energy level of the one-dimensional quantum harmonic oscillator. 

"""
#importing necessary libraries
import math
import matplotlib.pyplot as plt
import numpy as np 

#function to return nth Hermite polynomial 
def H(n,x):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return 2*x*H(n-1,x) - 2*(n-1)*H(n-2,x)

#function to return factorial of n 
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n*factorial(n-1)

#function to return the wavefunction for given n
def psi(n,x):
    return (1/math.sqrt((2**n)*factorial(n)*math.sqrt(math.pi))) * math.exp(-(x**2)/2) * H(n,x)

#calculating wavefunction values for x ranging from -4 and 4 and n = 0,1,2,3
x_range = np.arange(-4,4, 0.01)
psi_0_range = np.empty(shape = (0, ))
psi_1_range = np.empty(shape = (0, ))
psi_2_range = np.empty(shape = (0, ))
psi_3_range = np.empty(shape = (0, ))

for i in range(x_range.size):
    psi_0_range = np.append(psi_0_range, psi(0, x_range[i]))
    psi_1_range = np.append(psi_1_range, psi(1, x_range[i]))
    psi_2_range = np.append(psi_2_range, psi(2, x_range[i]))
    psi_3_range = np.append(psi_3_range, psi(3, x_range[i]))

#plotting wavefunction for given ranges of x and n
plt.plot(x_range, psi_0_range, label = "n=0")
plt.plot(x_range, psi_1_range, label = "n=1")
plt.plot(x_range, psi_2_range, label = "n=2")
plt.plot(x_range, psi_3_range, label = "n=3")
plt.legend()
plt.title('Wavefunction of nth Enery Level of a One-Dimensional Quantum Harmonic Oscillator')
plt.xlabel('x(m)')
plt.ylabel('$\Psi(n,x)$')
plt.show()


