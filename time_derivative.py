import numpy as np

def time_derivative(X, dt, type):
    m, n = np.shape(X)
    if(type == 1):
        roc = np.zeros((m,n))
        
        roc = (X[1:m,:] - X[0:m-1,:]) / dt # forward Euler
        roc[m-1,:] = (X[m-1,:] - X[m-2,:]) / dt # backward
    
    elif(type == 2):
        roc = np.zeros((m,n))
        
        roc[0,:] = (X[1,:] - X[0,:]) / dt
        roc[1:m-1,:] = (X[2:m,:] - X[0:m-2,:]) / (2*dt)
        roc[m-1,:] = (X[m-1,:] - X[m-2,:]) / dt
    
    return roc
