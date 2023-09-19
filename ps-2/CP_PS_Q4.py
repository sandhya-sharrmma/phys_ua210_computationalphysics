import math 
import numpy as np
import random

a = np.float128(0.001); b = np.float128(1000); c = np.float128(0.001)

#finding roots using first method 
x1_1 = np.float128((-b + math.sqrt((b**2) - (4*a*c)))/(2*a))
x2_1 = np.float128((-b - math.sqrt((b*b) - (4*a*c)))/(2*a))

print("Roots by using first method: ")
print("Root 1: ", x1_1)
print("Root 2: ", x2_1)
print()

#finding roots using second method 
x1_2 = np.float128((2*c)/(-b - math.sqrt((b*b) - (4*a*c))))
x2_2 = np.float128((2*c)/(-b + math.sqrt((b*b) - (4*a*c))))

print("Roots by using second method: ")
print("Root 1: ", x1_2)
print("Root 2: ", x2_2)
print()

print("The difference occurs due to approximation error and rounding off error since we are working with a big number like 1000 and small number like 0.001.")
print()

#comparing the roots given by both methods to the true roots given by Wolframalpha
x1_true = -1*(10**6)
x2_true = -1*(10**-6)

if abs(x1_1 - x1_true) < abs(x1_2 - x1_true):
    x1 = x1_1
else:
    x1 = x1_2

if abs(x2_1 - x2_true) < abs(x2_2 - x2_true):
    x2 = x2_1
else:
    x2 = x2_2

print("True roots given by the computation: ")
print("Root 1: ", x1)
print("Root 2: ", x2)

