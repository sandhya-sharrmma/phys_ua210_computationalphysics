"""
Name: CP_PS5_Q2.py
Author: Sandhya Sharma
Date: October 19, 2023
Description: Fitting datasets using polynomial and trigonometric functions. 

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as linalg

#retrieving data from input file and storing them in arrays
data = np.genfromtxt('/Users/sandhyasharma/Desktop/CompPhys/signal.dat', delimiter = "|", skip_header = 1)
time = data[:, 1]
signal = data[:, 2]

#plotting the data
plt.scatter(time, signal, color = 'midnightblue', label = 'data')
plt.xlabel("Time (s)")
plt.ylabel("Signal (V)")
plt.title("Signal vs Time")
plt.show()

#defining the polynomial fit function
def polynomial_fit(order, x, y):
    """
    This function fits a polynomial of particular order to the data provided.

    :param x: array of independent variable
    :param y: array of dependent variable
    :param order: order of the polynomial to be fitted
    :return: y_model: array of dependent variable for the fitted polynomial
    """
    A = np.zeros((len(x), order+1))
    for i in range(order+1):
        if i == 0:
            A[:, i] = 1.
        else:
            A[:, i] = x**i
   
    u,w,vt = linalg.svd(A, full_matrices = False)
    w_inv = np.zeros(shape = (w.size,))

    for i in range(w.size):
        if w[i] < 10**(-10):
            w_inv[i] = w[i]
        else:
            w_inv[i] = 1./w[i]

    ainv = vt.transpose().dot(np.diag(w_inv)).dot(u.transpose())

    c = ainv.dot(y)

    y_model = A.dot(c)
    
    print("Condition number for polynomial fit of order" + str(order) + ": ", np.max(w_inv)/np.min(w_inv))

    return y_model 

#plotting the data and the fitted polynomial for order 3
third_order_ymodel = polynomial_fit(3, time, signal)
plt.scatter(time, signal, color = 'midnightblue', label = 'data')
plt.scatter(time, third_order_ymodel, color = 'maroon', label = 'third-order poly fit')
plt.xlabel("Time (s)")
plt.ylabel("Signal (V)")
plt.title("Data Fitting using Polynomial Function of Order 3")
plt.legend()
plt.show()

#plotting the data and the fitted polynomial for order 3 with residuals 
third_order_residuals = abs(signal - third_order_ymodel)
plt.scatter(time, signal, color = 'midnightblue', label = 'data')
plt.scatter(time, third_order_ymodel, color = 'maroon', label = 'third-order poly fit')
plt.scatter(time, third_order_residuals, color = 'darkgreen', label = 'residuals')
plt.xlabel("Time (s)")
plt.ylabel("Signal (V)")
plt.title("Data Fitting using Polynomial Function of Order 3 with Residuals")
plt.legend()
plt.show()

#plotting the data and the fitted polynomial for order 8
eigth_order_ymodel = polynomial_fit(8, time, signal)
plt.scatter(time, signal, color = 'midnightblue', label = 'data')
plt.scatter(time, eigth_order_ymodel, color = 'darkorange', label = 'eigth order poly fit')
plt.xlabel("Time (s)")
plt.ylabel("Signal (V)")
plt.title("Data Fitting using Polynomial Function of Order 8")

#plotting the data and the fitted polynomial for order 8 with residuals 
eigth_order_residuals = abs(signal - eigth_order_ymodel)
plt.scatter(time, eigth_order_residuals, color = 'darkgreen', label = 'residuals')
plt.xlabel("Time (s)")
plt.ylabel("Signal (V)")
plt.title("Data Fitting using Polynomial Function of Order 8 with Residuals")
plt.legend()
plt.show()

#defining the trigonometric fit function
def trig_fit(n, q, x, y):
    
    """
    This function fits a fourier series to the data provided.

    :param x: array of independent variable
    :param y: array of dependent variable
    :param n: number of terms in the fourier series
    :param q: scale factor for the fundamental frequency of the fourier series
    :return: y_model: array of dependent variable for the fitted trigonometric function
    """
     
    f = q*(np.max(x)) #scaling the fundamental frequency by q
    A = np.zeros((len(x), 2*n+1))
    A[:, 0] = 1

    for i in range(1, n+1):
        A[:, 2*i-1] = np.cos(2.*np.pi*i*x/f)
        A[:, 2*i] = np.sin(2.*np.pi*i*x/f)
    
    u,w,vt = linalg.svd(A, full_matrices = False)
    w_inv = np.zeros(shape = (w.size,))

    for i in range(w.size):
        if w[i] == 0.:
            w_inv[i] = 0.
        else:
            w_inv[i] = 1./w[i]

    ainv = vt.transpose().dot(np.diag(w_inv)).dot(u.transpose())

    c = ainv.dot(y)

    y_model = A.dot(c)

    return y_model

#plotting the data and the fitted trigonometric function for n = 10, q = 0.5
trig_ymodel_half = trig_fit(10, 0.5, time, signal)
plt.scatter(time, signal, color = 'midnightblue', label = 'data')
plt.scatter(time, trig_ymodel_half, color = 'maroon', label = 'n = 10, q = 0.5')
plt.xlabel("Time (s)")
plt.ylabel("Signal (V)")
plt.title("Data Fitting using Trigonometric Functions")
plt.legend()
plt.show()

#plotting the data and the fitted trigonometric function for n = 10, q = 2
trig_ymodel_2 = trig_fit(10, 2, time, signal)
plt.scatter(time, signal, color = 'midnightblue', label = 'data')
plt.scatter(time, trig_ymodel_2, color = 'darkorange', label = 'n = 10, q = 2')
plt.xlabel("Time (s)")
plt.ylabel("Signal (V)")
plt.title("Data Fitting using Trigonometric Functions")
plt.legend()
plt.show()

#calculating residuals for the trigonometric fits
residuals_half = abs(signal - trig_ymodel_half)
residuals_2 = abs(signal - trig_ymodel_2)

#calculating r^2 for the trigonometric fits
ss_res_half = np.sum(residuals_half**2)
ss_tot_half = np.sum((signal - np.mean(signal))**2)
r_squared_half = 1 - (ss_res_half/ss_tot_half)

ss_res_2 = np.sum(residuals_2**2)
ss_tot_2 = np.sum((signal - np.mean(signal))**2)
r_squared_2 = 1 - (ss_res_2/ss_tot_2)

print()
print("r_sq value for fit = 1/2(time period): ", r_squared_half)
print("r_sq value for fit = 2(time period): ", r_squared_2)
print("Mean residual value for fit = 1/2(time period): ", np.mean(residuals_half))
print("Mean residual value for fit = 2(time period): ", np.mean(residuals_2))
print()





