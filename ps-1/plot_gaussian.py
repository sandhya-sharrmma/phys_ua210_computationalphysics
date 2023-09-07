import matplotlib.pyplot as plt
import numpy as np 

mean = 0
std = 3

values = np.random.normal(loc = mean, scale = std, size = 100000)

plt.ylabel("Probability")
plt.title("Gussian Curve")
plt.hist(values, range = (-10,10), bins = 500, density = True)
plt.savefig('gaussian.png')
plt.show()
