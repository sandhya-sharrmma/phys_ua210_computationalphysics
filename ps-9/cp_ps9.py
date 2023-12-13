'''
Name: cp_ps9.py
Author: Sandhya Sharma
Description: This program solves the time-dependent Schrodinger equation using Cranck-Nicholson method.
Date: December 12, 2023

'''

import numpy as np 
import matplotlib.pyplot as plt
import banded 

#constants 
h_bar = 1.0545718e-34 #J*s
m = 9.10938356e-31 #kg
L = 10**(-8) #length of box
N = 1000 #number of steps
a = L/N #step size
h = 10**(-18) #time step
i = complex(0,1) #imaginary number

#coefficients for the tridiagonal matrix
a1 = 1 + h*(i*h_bar/(2*m*a**2))
a2 = -h*(i*h_bar/(4*m*a**2))
b1 = 1 - h*(i*h_bar/(2*m*a**2))
b2 = h*(i*h_bar/(4*m*a**2))

#initial wavefunction
def psi_0(x):
    '''
    This function returns the initial wavefunction at time t = 0.

    Parameters:
    x : position

    Returns:
    psi : initial wavefunction
    '''
    sigma = 10**(-10)
    k = 5*(10**(10))
    return np.exp(-((x-L/2)**2)/(2*(sigma**2)))*np.exp(i*k*x)

#initializing the A matrix
A = np.zeros((3,N),dtype=complex)
A[0, :] = a2
A[1, :] = a1
A[2, :] = a2

#initializing the wavefunction
x = np.linspace(0,L,N+1)
psi = np.zeros((N+1),dtype=complex)
psi[:] = psi_0(x)
psi[[0,N]] = 0

#arrays to store results for plotting
time_values = np.arange(0, N*h, h)
wave = np.zeros((N,N+1),dtype=complex)

#solving the banded matrix using banded.py by Mark Newman 
for n in range(time_values.size):
    wave[n,:] = psi
    v = b1*psi[1:N] + b2*(psi[2:N+1] + psi[0:N-1])
    psi[1:N] = banded.banded(A,v,1,1)

#taking the real part of the wavefunction
wave = np.real(wave)

#plotting the results
fig = plt.figure(figsize=(10, 20)) 
for s in range(4):
    plt.subplot(2, 2, s+1)  
    plt.plot(x, wave[250*s, :])
    plt.xlabel('x (m)')
    plt.ylabel('Re[psi(x)]')
fig.suptitle('Wavefunction vs. Position')
plt.show()
