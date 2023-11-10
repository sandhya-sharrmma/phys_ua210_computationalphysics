"""
Name: CP_PS7_Q1.py
Author: Sandhya Sharma
Date: November 9, 2023
Description: Finding the minimum of a function using Brent's method and SciPy's method.

"""

import math 
import numpy as np
import scipy.optimize 

def f(x):
    '''
    The function returns the function of which the minimum is to be found.

    Parameters: 
    x : The value at which the function is to be evaluated.

    Returns:
    The value of the function at x.
    '''
    return ((x-0.3)**2)*np.exp(x)

def s_quad_interpolation(a, b, c):
    """
    The function computes the inverse quadratic interpolation.

    Parameters:
    a : The first point.
    b : The second point.
    c : The third point.

    Returns:
    The value of the inverse quadratic interpolation.
    """
    e = 10**-7 #for numerical stability
    s0 = a*f(b)*f(c) / (e + (f(a)-f(b))*(f(a)-f(c)))
    s1 = b*f(a)*f(c) / (e + (f(b)-f(a))*(f(b)-f(c)))
    s2 = c*f(a)*f(b) / (e + (f(c)-f(a))*(f(c)-f(b)))
    return s0+s1+s2

def brent(f, a, b, tol = 10**-7):
    """
    The function implements Brent's method to find the minimum of a function.

    Parameters:
    f: The function of which the minimum is to be found.
    a : Lower limit of the interval.
    b : Upper limit of the interval.
    tol : The tolerance value for the difference between the two points.

    Returns:
    The value of the inverse quadratic interpolation.
    """
    if abs(f(a)) < abs(f(b)):
        a, b = b, a

    c = a
    u = True 

    while abs(b-a) > tol:
        if f(a) != f(c) or f(b) != f(c):
            s = s_quad_interpolation(a, b, c)
        elif f(a) != f(b):
            s = (b*f(a) - a*f(b))/(f(a) - f(b))

        if (s-(3*a+b)/4)*(s-b) > 0 or (u and abs(s-b) >= abs(b-c)/2) or (not u and abs(s-b) >= abs(c-d)/2) or (u and abs(b-c) < tol) or (not u and abs(c-d) < tol):
            # s = golden_mean_search(f, a, b) #alternative method to bisection
            s = (a+b)/2
            u = True
        else:
            u = False
        
        d = c
        c = b

        if f(a)*f(s) < 0:
            b = s
        else:
            a = s
        
        if abs(f(a)) < abs(f(b)):
            a, b = b, a

    return b

def golden_mean_search(f, a,b, tol = 10**-7):
    """
    The function implements the golden mean search method to find the minimum of a function.

    Parameters:
    f: The function of which the minimum is to be found.
    a : Lower limit of the interval.
    b : Upper limit of the interval.
    tol : The tolerance value for the difference between the two points. 

    Returns:
    The value of the inverse quadratic interpolation.
    """
    golden_ratio = (1 + math.sqrt(5))/2

    c = b - (b-a)/golden_ratio
    d = a + (b-a)/golden_ratio

    while abs(b-a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b-a)/golden_ratio
        d = a + (b-a)/golden_ratio
    
    return (a+b)/2

#printing the results 
print()
print("Using Brent's method:")
print(brent(f, -1, 5))
print()
print("Using SciPy's method: ")
print(scipy.optimize.brent(f, brack=(-1, 5)))
print()
