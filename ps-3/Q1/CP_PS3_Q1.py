"""
Name: CP_PS3_Q1.py
Author: Sandhya Sharma
Date: September 21, 2023
Description: Performing derivatives on a function as the limit of delta goes to zero and comparing their accuracy to the true value of differentiation. 

"""

#importing necessary libraries 
import numpy as np 
import matplotlib.pyplot as plt
import math

#defining the function
def func(x): 
    return x*(x-1)

#function to calculate derivative computationally 
def derivative(x,delta):
    f_x = (func(x+delta) - func(x))/delta
    return f_x

#setting the values for delta and x
delta = 10**(-2)
x = 1 

#calculating the derivative computationally and analytically 
func_x = derivative(x, delta)
func_x_true = (2*x) - 1

print()
print("Function: f(x) = x(x-1)")
print("At x = " + str(x) + "," + " Delta = " + str(delta))
print("df/dx = ", func_x)
print("True value of derivative (analytically) = ", func_x_true)
print("Fractional difference = ", (func_x - func_x_true)/func_x_true)
print()

#caluclating derivative with different values for delta 
delta_array = np.array([10**-4, 10**-6, 10**-8, 10**-10, 10**-12, 10**-14])
func_x_array = np.empty(shape = (0,))

for i in range(delta_array.size):
    func_x = derivative(x, delta_array[i])
    func_x_array = np.append(func_x_array, func_x)
    print("Delta = " + str(delta_array[i]) + ", df/dx = " + str(func_x_array[i]))

error = func_x_array - func_x_true
error = np.abs(error)

print()

#plotting error vs log10(delta)
plt.scatter(np.log10(delta_array), error)
plt.xlabel('log10(delta) (no units)')
plt.ylabel('Error (no units)')
plt.title('Error v/s log10(delta)')
plt.savefig('cp_ps3_q1.png')
plt.show()
