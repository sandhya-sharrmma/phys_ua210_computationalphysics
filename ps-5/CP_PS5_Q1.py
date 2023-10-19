"""
Name: CP_PS5_Q1.py
Author: Sandhya Sharma
Date: October 19, 2023
Description: Finding the value of gamma function for different values of a using 
             the method of Gaussian quadrature.
             
"""

import matplotlib.pyplot as plt
import math 
import numpy as np
from numpy import ones,copy,cos,tan,pi,linspace
import scipy.integrate as integrate

def gamma_integrand(a,x):
    return (x**(a-1))*math.exp(-x)

x_range = np.arange(0, 5, 0.1) 
gamma_2_range = np.empty(shape = (0,)) 
gamma_3_range = np.empty(shape = (0,)) 
gamma_4_range = np.empty(shape = (0,)) 

for i in range(x_range.size):
    gamma2 = gamma_integrand(2, x_range[i])
    gamma3 = gamma_integrand(3, x_range[i])
    gamma4 = gamma_integrand(4, x_range[i])
    gamma_2_range = np.append(gamma_2_range, gamma2)
    gamma_3_range = np.append(gamma_3_range, gamma3)
    gamma_4_range = np.append(gamma_4_range, gamma4)

plt.plot(x_range, gamma_2_range, label = "a = 2")
plt.plot(x_range, gamma_3_range, label = "a = 3")
plt.plot(x_range, gamma_4_range, label = "a = 4")
plt.legend()
plt.xlabel('x')
plt.ylabel("gamma(x)")
plt.title("Gamma Distribution for Different Values of a")

def gamma_integrand_cov(a,x):
    return np.exp((a-1)*np.log(x)-x) 

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

#integrating by using the method of Gaussian quadrature 
a = 0 #lower bound for integration after rescaling
b = 1 #upper bound for integration after rescaling 
p = 100 #number of points used for Gaussian quadrature 

#getting the weights and roots
x, w = gaussxw(p)
xp = 0.5*(b-a)*x + 0.5*(b+a)
wp = 0.5*(b-a)*w

#adding to the value of integration for every point 
def gamma(a): 
    integration = 0; 
    for i in range(p):
        z = xp[i]
        integration += wp[i] * gamma_integrand_cov(a, ((a-1)*z)/(1-z)) * (a-1)/((1-z)**2) #rescaling the integral from (0, inf) to (0,1)
    return integration



print()
print("Values for gamma functions: ")
print("gamma(1.5) = ", gamma(1.5))
print("gamma(3) = ", gamma(3))
print("gamma(6) = ", gamma(6))
print("gamma(10) = ", gamma(10))
print()
print("Comparing with the factorials: ")
print("gamma(3) = ", math.factorial(2))
print("gamma(6) = ", math.factorial(5))
print("gamma(10) = ", math.factorial(9))
print()

plt.show()





