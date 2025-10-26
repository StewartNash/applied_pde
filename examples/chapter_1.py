
## Problem 1.2.2

import numpy as np
import matplotlib.pyplot as plt

N = 10 # Length of rod
T = 5 # Duration of simulation

def matrix_power(x, n):
    y = x.copy()
    if n > 1:
        for i in np.arange(n - 1):
            y = np.matmul(y, x)
    return y

coefficients = np.zeros((N, N))
for i in np.arange(N):
    if i == 0:
        coefficients[i, 0]  = 1
    elif i == N - 1:
        coefficients[i, N - 1] = 1
    else:
        coefficients[i, i - 1] = 0.1
        coefficients[i, i] = 0.8
        coefficients[i, i + 1] = 0.1

initial_conditions = np.zeros((N, 1))
initial_conditions[0, 0] = 1
initial_conditions[N - 1, 0] = -1
y = [np.squeeze(np.transpose(np.matmul(matrix_power(coefficients, i), initial_conditions))) for i in np.arange(T)]

length_intervals = np.arange(N)
plt.figure()
for i in np.arange(T):
    plt.plot(length_intervals, y[i])
plt.show()
print("The number of non-zero elements is {0}.".format(np.count_nonzero(y[T - 1] != 0)))

