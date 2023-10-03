"""
Name: CP_PS4_Q3cd.py
Author: Sandhya Sharma
Date: October 3, 2023
Description: Finding the quantum uncertainty in the position of a particle in the nth level of a harmonic oscillator using Gaussian quadrature, Gauss-Hermite quadrature and SciPy quad function. 

"""

#importing necessary libraries
import math
import matplotlib.pyplot as plt
import numpy as np 
from numpy import ones,copy,cos,tan,pi,linspace
import scipy 
from scipy.integrate import quad

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

#function to return the integrand required to compute the quantum uncertainty
def f(x):
    return x**2 * psi(5,x)**2

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

#integrating by using the method of Gaussian quadrature 
a = -1 #lower bound for integration after rescaling
b = 1 #upper bound for integration after rescaling 
p = 100 #number of points used for Gaussian quadrature 

#getting the weights and roots
x, w = gaussxw(p)
xp = 0.5*(b-a)*x + 0.5*(b+a)
wp = 0.5*(b-a)*w

#adding to the value of integration for every point 
integration = 0; 
for i in range(p):
    z = xp[i]
    integration += wp[i] * f(z/(1-(z**2))) * (1+z**2)/(1-z**2)**2 #rescaling the integral from (-inf, inf) to (-1,1)

uncertainty = math.sqrt(integration)

print()
print("Quantum uncertainty using Gaussian-quadrature: ", uncertainty)

#integration using Gauss-Hermite quadrature 

#retrieving n weights and roots 
xpoints_hermgauss, weights_hermgauss = np.polynomial.hermite.hermgauss(p)
xp_hermgauss = 0.5*(b-a)*xpoints_hermgauss + 0.5*(b+a)
wp_hermgauss = 0.5*(b-a)*weights_hermgauss

#calculating the integral 
integration_hermgauss = 0
for i in range(p):
    z = xp_hermgauss[i]
    integration_hermgauss += weights_hermgauss[i] * f(z/(1-(z**2))) * (1+z**2)/(1-z**2)**2 #rescaling the integral from (-inf, inf) to (-1,1)

#scaling the result by the normalization factor
integration_hermgauss *= (math.pi) 

print("Quantum uncertainty using Gauss-Hermite quadrature: ", math.sqrt(integration_hermgauss)) 

#using scipy's quad to find the exact integral
quad_result, _ = quad(f, -np.inf, np.inf)
print("Quantum uncertainty using SciPy's quad function:", math.sqrt(quad_result))
print()








