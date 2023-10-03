"""
Name: CP_PS4_Q1.py
Author: Sandhya Sharma
Date: September 28, 2023
Description: Integrating a given function and estimating its error. 

"""

#function definition to return f(x)
def f(x):
    return x**4 - 2*x + 1

#defining parameters for trapezium rule
n1 = 10; n2 = 20 #number of steps 
a = 0; b = 2 #lower and upper bounds
h1 = (b-a)/n1; h2 = (b-a)/n2 #width of each step 

#initializing sum of f(x) values
s1 = 0.5*(f(a) + f(b))
s2 = s1

#adding to the values of f(x) with each step 
for i in range(1, n1):
    s1 += f(a + (i*h1))

for i in range(n1, n2):
    s2 += f(a + (i*h2))

#multipying each sum with width value to get the integral values 
i1 = s1*h1; i2 = s2*h2 

#estimating error on integral
e2 = 1/3*abs(i2-i1)

#printing results in the console 
print()
print("f(x) = x^4 - 2x + 1")
print("Integral of f(x) using N = 10: ", i1)
print("Integral of f(x) using N = 20: ", i2)
print("Estimated error              :",  e2)
print()
