import numpy as np
import matplotlib.pyplot as plt

x_array = np.arange(-200, 200, 0.1)

def h_function(arr):
    return (0.1 * np.arctan(arr))/np.arctan(2)

h_array = h_function(x_array)

plt.plot(x_array, h_array)
plt.hlines(0, ls="--", color = "gray", xmin = -205, xmax = 205)
plt.title("Plot of h(x̂)", fontsize = 30)
plt.xlabel("x̂", fontsize = 15)
plt.ylabel("h(x̂)", fontsize = 15)
plt.grid()
plt.show()