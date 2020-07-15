  
import numpy as np 

def lorenz96(t, x, n, F):
    # used periodicity condition: x_n = x_0, x_{n+1} = x_1
    xdot = np.zeros(n)
    xdot[0] = - x[n-2] * x[n-1] + x[n-1]* x[1] - x[0] + F
    xdot[1] = - x[n-1] * x[0] + x[0] * x[2] - x[1] + F
    xdot[n-1] = - x[n-3] * x[n-2] + x[n-2] * x[0] - x[n-1] + F

    for i in range(2, n-1):
        xdot[i] = - x[i-2] * x[i-1] + x[i-1] * x[i+1] - x[i] + F # lorenz 96 dx_i/dt equation

    return xdot