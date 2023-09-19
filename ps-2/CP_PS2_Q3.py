import math 
import numpy as np 
import matplotlib.pyplot as plt

#setting the range for real and complex values 
x = np.arange(-2,2.1, 0.1)
y = np.arange(-2,2.1, 0.1)

#creating empty numpy arrays for z and c 
z_array = np.empty(shape = (0,), dtype=complex)
c_array = np.empty(shape = (0,), dtype=complex)

#iterating over the Mandelbrot equation for every value of c and storing the values in z_array
iterations = 1000
for i in range(x.size):
    for j in range(y.size):
        c = complex(x[i], y[j])
        c_array = np.append(c_array, c)
        z = 0 + 1j*0
        for n in range(iterations):
            if np.abs(z) > 2:
                break
            z = z*z + c
        z_array = np.append(z_array, z)

#finding the magnitude of z for every value of c
z_magnitudes = np.abs(z_array)

#storing z values in two different arrays for points in and out of Mandelbrot set
c_in = np.empty(shape = (0,), dtype=complex)
c_out = np.empty(shape = (0,), dtype=complex)

for i in range(z_magnitudes.size):
    if z_magnitudes[i] > 2:
        c_in = np.append(c_in, c_array[i])
    else:
        c_out = np.append(c_out, c_array[i])

#plotting the Mandelbrot set 
plt.scatter(np.real(c_in), np.imag(c_in), color = 'black')
plt.scatter(np.real(c_out), np.imag(c_out), color = 'white')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig('cp_ps2_q2.png')
plt.show()











        
