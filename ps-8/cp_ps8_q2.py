"""
Name: cp_ps8_q2.py
Author: Sandhya Sharma
Date: November 30, 2023
Description: Solving the Lorentz equations using solve_ivp of scipy.integrate. 

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Defining the Lorentz function
def lorentz_equations(t, y, sigma, rho, beta):
    x, y, z = y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

#Assigning parameters
sigma = 10.0
rho = 28.0
beta = 8.0/3.0

#Initial conditions
y0 = [0.0, 1.0, 0.0]

#Specifying time span
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000) #t_eval is the time points at which the solution is to be computed

#Solving the equations using solve_ivp of scipy.integrate
sol = solve_ivp(lorentz_equations, t_span, y0, t_eval = t_eval, args=(sigma, rho, beta))

#Plotting the solutions
plt.plot(sol.t, sol.y[1])
plt.title('y(t) v/s t')
plt.xlabel('t (s)')
plt.ylabel('y(t) (no units)')
plt.show()

p = plt.scatter(sol.y[0], sol.y[2], s=5.0, c = sol.t, cmap = 'bone')
plt.title('z(t) v/s x(t): The Strange Attractor')
plt.xlabel('x(t) (no units)')
plt.ylabel('z(t) (no units)')
plt.colorbar(p)
plt.show()