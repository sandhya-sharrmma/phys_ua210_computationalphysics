import math 
import numpy as np 
import timeit

start1 = timeit.default_timer()
#function to calculate potential due to single atom at distance d 
def potential(i, j, k):
    d = float(math.pow((i**2) + (j**2) + (k**2), 0.5))
    potential = (math.pow(-1, (i+j+k)%2))/d
    return potential 

#fixing the number of atoms in x, y and z direction 
L = 100

#using a for-loop
potential_sum = 0; 

for i in range (-1*L, L+1):
    for j in range(-1*L, L+1):
        for k in range(-1*L, L+1):
            if i == 0 and j == 0 and k == 0:
                continue
            potential_sum += potential(i,j,k)

print("By using a for loop:")
print("The value of Madelung constant is " + str(potential_sum))

#finding runtime for using a loop
stop1 = timeit.default_timer()
print("Runtime for using a loop (s): ", stop1 - start1)
print()

#without using a for-loop
start2 = timeit.default_timer()

i = np.arange(-L, L + 1)
j = np.arange(-L, L + 1)
k = np.arange(-L, L + 1)

#calculate the sum array
sum_array = i[:, np.newaxis, np.newaxis] + j[np.newaxis, :, np.newaxis] + k[np.newaxis, np.newaxis, :]
sum_array = sum_array.flatten()

#calculate the distance array
i_squared = np.power(i, 2)
j_squared = np.power(j, 2)
k_squared = np.power(k, 2)

distance_array = i_squared[:, np.newaxis, np.newaxis] + j_squared[np.newaxis, :, np.newaxis] + k_squared[np.newaxis, np.newaxis, :]
distance_array = distance_array.flatten()
distance_array = np.sqrt(distance_array)

#finding the index where distance = 0 and removing it from distance_array and sum_array
index_to_be_removed = np.argwhere(distance_array == 0)
index_to_be_removed = index_to_be_removed[0][0]

distance_array = np.delete(distance_array, index_to_be_removed)
sum_array = np.delete(sum_array, index_to_be_removed)

#calculating the individiual potential due to particular atoms and summing them
potential = np.power(-1, sum_array%2)/distance_array
result = np.sum(potential)

print("Without using a for loop:")
print("The value of Madelung constant is " + str(result))

#finding runtime without using a loop
stop2 = timeit.default_timer()
print("Runtime without using a loop (s): ", stop2 - start2)
print()

#determing which method is faster 
if (stop2 - start2) < (stop1 - start1):
    print("Program without a for-loop is faster.")
else:
    print("Program with for-loop is faster.")
    print()



















