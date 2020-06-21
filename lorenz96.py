import numpy as np 

def lorenz96(t, x, n, F):
    xdot = np.zeros((n, 1))
    xdot[0,] = - x[n-2,] * x[n-1,] + x[n-1,]* x[1,] - x[0,] + F
    xdot[1,] = - x[n-1,] * x[0,] + x[0,] * x[2,] - x[1,] + F
    xdot[n-1,] = - x[n-3] * x[n-2,] + x[n-2,] * x[0] - x[n-1,] + F

    for i in range(2, n-1):
        xdot[i,] = - x[i-2,] * x[i-1] + x[i-1,] * x[i+1,] - x[i] + F
    
    return xdot