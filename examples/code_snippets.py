
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


## Problem 8.3.4

import numpy as np

# Forward Difference Method - Dirichlet Initial-Boundary-Value Problem
def algorithm_8_1(diffusivity,
                  endpoint,
                  time_step,
                  number_of_time_steps,
                  number_of_nodes,
                  right_side,
                  initial_condition,
                  boundary_condition_left,
                  boundary_condition_right):
    # Define a grid
    increment = endpoint / (number_of_nodes + 1)
    coefficient_r = diffusivity * time_step / increment ** 2
    if coefficient_r > 0.5:
        print("WARNING: algorithm_8_1 is unstable")
        
    # Initialize numerical solution
    t = np.zeros((number_of_time_steps + 1,))
    #x = np.zeros((1, number_of_nodes + 2))
    x = np.zeros((number_of_nodes + 2,))
    x[0] = 0
    #V = np.zeros((1, number_of_nodes + 2))
    V = np.zeros((number_of_nodes + 2,))
    V[0] = (boundary_condition_left(0) + initial_condition(0)) / 2
    for n in np.arange(number_of_nodes):
        x[n + 1] = x[n] + increment
        V[n + 1] = initial_condition(x[n + 1])
    x[number_of_nodes + 1] = endpoint
    V[number_of_nodes + 1] = (boundary_condition_right(0) + initial_condition(endpoint)) / 2

    # Begin time stepping
    #U = np.zeros((1, number_of_nodes + 2))
    U = np.zeros((number_of_nodes + 2,))    
    for j in np.arange(number_of_time_steps):
        # Advance solution one time step
        for n in np.arange(number_of_nodes):
            U[n + 1] = coefficient_r * V[n]
            U[n + 1] += (1 -  2 * coefficient_r) * V[n + 1]
            U[n + 1] += coefficient_r * V[n + 2]
            U[n + 1] += time_step * right_side(x[n + 1], t[j])
        t[j + 1] = t[j]  + time_step
        U[0] = boundary_condition_left(t[j + 1])
        U[number_of_nodes + 1] = boundary_condition_right(t[j + 1])        
        # Output numerical solution
        # Prepare for next time step
        for n in np.arange(number_of_nodes + 2):
            V[n] = U[n]
            
    #x = x[1:-1]
    t = t[:-1]
            
    return U, x, t

import math

# right_side
def S(x, t):
    return -2.0 * math.e ** (x - t)

# initial_condition
def f(x):
    return math.e ** x

# boundary_condition_left
def p(t):
    return  math.e ** -t

# boundary_condition_right
def q(t):
    return math.e ** (1 - t)

# exact_answer	
def u(x, t):
	return math.e ** (x - t)

a2 = 1 # diffusivity
L = 1 # endpoint
k = 0.0025 # time_step
nmax = 9 # number_of_nodes
end_time = 0.5
jmax = int(end_time / k) # number_of_time_steps

numerical_answer, x, t = algorithm_8_1(a2,
                       L,
                       k,
                       jmax,
                       nmax,
                       S,
                       f,
                       p,
                       q)
exact_answer = 	u(x, t[-1])
answer_error = (numerical_answer - exact_answer) / (exact_answer)
answer_error = answer_error * 100

from matplotlib import pyplot as plt

fig, ax1 = plt.subplots()

ax1.set_xlabel('Position')
ax1.set_ylabel('Temperature')
ax1.plot(x, numerical_answer, 'r', label='Numerical Solution')
ax1.plot(x, exact_answer, 'g', label='Exact Solution')

ax2 = ax1.twinx()
ax2.set_ylabel('Percent Error')
ax2.plot(x, answer_error, 'b', label='Percent Error')

ax1.legend()
ax2.legend()

plt.show()

## Problem 8.3.5

import numpy as np


# Solution of a Tridiagonal Linear System
def algorithm_8_2(a, # subdiagonal
                 b, # diagonal
                 c, # superdiagonal
                 d, # right-hand side
                 number_of_nodes=None):
    if number_of_nodes is None:
        number_of_nodes = d.size
    # Forward substitute to eliminate subdiagonal
    for n in np.arange(number_of_nodes - 1):
        ratio = a[n + 2] / b[n + 1]
        b[n + 2] = b[n + 2] - ratio * c[n + 1]
        d[n + 2] = d[n + 2] - ratio * d[n + 1]
    # Back substitude and store in solution array in d
    d[number_of_nodes] = d[number_of_nodes] / b[number_of_nodes]
    for l in np.arange(number_of_nodes - 1):
        n = number_of_nodes - l
        d[n] = (d[n] - c[n] * d[n + 1]) / b[n]
    return d
        
# Backward Difference Method - Dirichlet Initial-Boundary-Value Problem
def algorithm_8_3(diffusivity,
                  endpoint,
                  time_step,
                  number_of_time_steps,
                  number_of_nodes,
                  right_side,
                  initial_condition,
                  boundary_condition_left,
                  boundary_condition_right):
    # Define a grid
    increment = endpoint / (number_of_nodes + 1)
    coefficient_r = diffusivity * time_step / increment ** 2
        
    # Initialize numerical solution
    t = np.zeros((number_of_time_steps + 1,))
    x = np.zeros((number_of_nodes + 2,))
    x[0] = 0
    U = np.zeros((number_of_nodes + 2,))
    U[0] = (boundary_condition_left(0) + initial_condition(0)) / 2
    for n in np.arange(number_of_nodes):
        x[n + 1] = x[n] + increment
        U[n + 1] = initial_condition(x[n + 1])
    x[number_of_nodes] = endpoint
    U[number_of_nodes + 1] = (boundary_condition_right(0) + initial_condition(endpoint)) / 2

    term_a = np.zeros((number_of_nodes + 2,))
    term_b = np.zeros((number_of_nodes + 2,))
    term_c = np.zeros((number_of_nodes + 2,))
    term_d = np.zeros((number_of_nodes + 2,))    
    # Begin time stepping  
    for j in np.arange(number_of_time_steps):
        # Define tridiagonal system
        t[j + 1] = t[j]  + time_step
        for n in np.arange(number_of_nodes):
            term_a[n + 1] = - coefficient_r
            term_b[n + 1] = 1 + 2 * coefficient_r
            term_c[n + 1] = - coefficient_r
            term_d[n + 1] = U[n + 1] + time_step * right_side(x[n + 1], t[j + 1])
        term_d[1] = term_d[1] + coefficient_r * boundary_condition_left(t[j + 1])
        term_d[number_of_nodes] = term_d[number_of_nodes] + coefficient_r * boundary_condition_right(t[j + 1])
        # Advance solution one time step
        term_d = algorithm_8_2(term_a, term_b, term_c, term_d, number_of_nodes)
        for n in np.arange(number_of_nodes):
            U[n + 1] = term_d[n + 1]
        U[0] = boundary_condition_left(t[j + 1])        
    # Output numerical solution        
    U[0] = boundary_condition_left(t[j + 1])
    U[number_of_nodes + 1] = boundary_condition_right(t[j + 1])

    t = t[:-1]
            
    return U, x, t

import math

# right_side
def S(x, t):
    return -2.0 * math.e ** (x - t)

# initial_condition
def f(x):
    return math.e ** x

# boundary_condition_left
def p(t):
    return  math.e ** -t

# boundary_condition_right
def q(t):
    return math.e ** (1 - t)

# exact_answer	
def u(x, t):
	return math.e ** (x - t)

a2 = 1 # diffusivity
L = 1 # endpoint
k = 0.0025 # time_step
nmax = 9 # number_of_nodes
end_time = 0.5
jmax = int(end_time / k) # number_of_time_steps

numerical_answer, x, t = algorithm_8_3(a2,
                       L,
                       k,
                       jmax,
                       nmax,
                       S,
                       f,
                       p,
                       q)
exact_answer = u(x, t[-1])
answer_error = (numerical_answer - exact_answer) / (exact_answer)
answer_error = answer_error * 100

from matplotlib import pyplot as plt

fig, ax1 = plt.subplots()

ax1.set_xlabel('Position')
ax1.set_ylabel('Temperature')
ax1.plot(x, numerical_answer, 'r', label='Numerical Solution')
ax1.plot(x, exact_answer, 'g', label='Exact Solution')

ax2 = ax1.twinx()
ax2.set_ylabel('Percent Error')
ax2.plot(x, answer_error, 'b', label='Percent Error')

ax1.legend()
ax2.legend()

plt.show()
