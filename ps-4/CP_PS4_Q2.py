"""
Name: CP_PS4_Q2.py
Author: Sandhya Sharma
Date: September 28, 2023
Description: Finding time period of anharmonic oscillation depending on its amplitude by using Gaussian quadrature. 

"""
#importing necessary libraries
import numpy as np 
from numpy import ones,copy,cos,tan,pi,linspace
import math
import matplotlib.pyplot as plt

#function to return the value of potential V(x)
def V(x):
    return x**4 

#function for finding the integration points and weights for Gaussian quadrature
#written by Mark Newman, June 4, 2011
def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

#function to return time period T(x) 
def t(x,m,a):
    t = math.sqrt((8*m)/(V(a) - V(x)))
    return t

m = 1 #mass 
a = 2 #amplitude
n = 20 #number of steps 

#setting arrays for different values of amplitide and their respective integration values
a_array = np.arange(0, a, 0.01) 
integration_array = np.empty(shape = (0, ))

#calculating integration values (time periods) for amplitude 'a' ranging from 0 to 2
for j in range(a_array.size): 
    x, w = gaussxw(n) 
    xp = 0.5*(a_array[j])*x + 0.5*(a_array[j])
    wp = 0.5*(a_array[j])*w

    integration = 0

    for i in range(n):
        integration += wp[i]*t(xp[i], m, a_array[j])
    
    integration_array = np.append(integration_array, integration)

#plotting time period against varying amplitude
plt.plot(a_array, integration_array, color = 'lightsteelblue') 
plt.xlabel("Amplitude (m)")
plt.ylabel("Time period (s)")
plt.title('Amplitude vs Time Graph for an Anharmonic Oscillator')
plt.savefig('cp_ps4_q1.png')
plt.show()






