"""
Name: CP_PS7_Q2.py
Author: Sandhya Sharma
Date: November 10, 2023
Description: Performing a logistic regression. 

"""

import matplotlib.pyplot as plt
import numpy as np
import csv 
import os
from scipy import optimize

#Reading the data from the csv file
filename = 'survey.csv'
curr_path = os.getcwd() + '/' + filename 

age = np.empty(shape = (0, ))
recognize = np.empty(shape = (0, ))

with open(curr_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        age = np.append(age, float(row['age']))
        recognize = np.append(recognize, float(row['recognized_it']))

sort = np.argsort(age)
age = age[sort]
recognize = recognize[sort]


def p(x, b0, b1):
    '''
        The function returns the probability of a logistic function. 

        Parameters:
        x : The value at which the function is to be evaluated.
        b0 : The first parameter.
        b1 : The second parameter.

        Returns:
        A logistic function.
    '''
    return 1/(1+np.exp(-b0 - b1*x))

def log_likelihood(beta, xs, ys):
    '''
        The function returns the negative log likelihood of the data. 

        Parameters:
        beta : The parameters of the logistic function.
        xs : The x values of the data.
        ys : The y values of the data.

        Returns:
        The negative log likelihood of the data.
    '''
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-p(xs[i], beta_0, beta_1)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return -ll 

#initial guess for the values of the parameters b0 and b1
beta = np.array([-0.01, 0.10])

#defining the error function
errfunc = lambda beta, x, y: log_likelihood(beta, x, y) - recognize

def Covariance(hess_inv, result_variance):
    '''
        The function returns the covariance matrix.

        Parameters:
        hess_inv : The inverse of the Hessian matrix.
        result_variance : The variance of the result.

        Returns:
        The covariance matrix.
    '''
    return hess_inv * result_variance

def error(hess_inv, result_variance):
    '''
        The function returns the error in the parameters.

        Parameters:
        hess_inv : The inverse of the Hessian matrix.
        result_variance : The variance of the result.

        Returns:      
        The error in the parameters.
    '''
    covariance = Covariance(hess_inv, result_variance)
    return np.sqrt(np.diag(covariance))

#performing the optimization
result = optimize.minimize(lambda beta, age, recognize: np.sum(errfunc(beta, age, recognize)), beta, args = (age, recognize))
hess_inv = result.hess_inv
result_variance = result.fun/(len(recognize)-len(beta))
error_value = error(hess_inv, result_variance)

#displaying the results
print()
print('Optimal parameters:')
print('b0 = ', result.x[0])
print('b1 = ', result.x[1])
print()
print('Errors:')
print('Error in b0 = ', error_value[0])
print('Error in b1 = ', error_value[1])
print()
print('Covariance Matrix:')
print('C = ', Covariance( hess_inv,  result_variance))
print()

x = np.arange(0, 85, 0.11)
plt.plot(x, p(x, result.x[0], result.x[1]), label = 'Logistic Regression')
plt.scatter(age, recognize, label = 'Data')
plt.xlabel('Age (years)')
plt.ylabel('p(x) (unitless)')
plt.title('Data Distribution according to Age')
plt.legend(loc = 'best')
plt.show()



