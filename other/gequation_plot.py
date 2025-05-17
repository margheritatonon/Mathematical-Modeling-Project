import numpy as np
import matplotlib.pyplot as plt

x_array = np.arange(-2, 10, 0.01)

def h_function(arr):
    return arr * (7-arr)

h_array = h_function(x_array)

plt.scatter(0, 0, color = "red", label = "zero")
plt.scatter(7, 0, color = "red", label = "lambda")
plt.plot(x_array, h_array)
plt.hlines(0, ls="--", color = "gray", xmin = -2, xmax = 10)
plt.title("Plot of g(n)", fontsize = 30)
plt.xlabel("n", fontsize = 15)
plt.ylabel("g(n)", fontsize = 15)
plt.grid()
plt.show()