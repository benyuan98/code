import numpy as np
    # calculates dx/dt for all inputs given a scheme
    # note input X looks like: x1(t0) ... xn(t0)
    #                          x1(t1) ... xn(t1)
    #                           ...
    #                          x1(t_{m-1}) ... xn(t_{m-1})
def time_derivative(X, dt, type): 
    m, n = np.shape(X)
    if(type == 1):
        roc = np.zeros((m,n))
        
        roc = (X[1:m,:] - X[0:m-1,:]) / dt # forward Euler
        roc[m-1,:] = (X[m-1,:] - X[m-2,:]) / dt # backward
    
    elif(type == 2):
        roc = np.zeros((m,n))
        
        roc[0,:] = (X[1,:] - X[0,:]) / dt # forward Euler
        roc[1:m-1,:] = (X[2:m,:] - X[0:m-2,:]) / (2*dt) # central
        roc[m-1,:] = (X[m-1,:] - X[m-2,:]) / dt # backward
    
    return roc