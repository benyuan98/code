import numpy as np

# note compared to MATLAB code, subtracted 1 from each index. Also used integer division to typecheck(shouldn't affect the result because the outcome should have been an integer in original code)
def Lorenz96_true_coefficients(n,F):
    N = (n+1)*(n+2)//2 # number of columns of dictionary
    cTrueMat = np.zeros((N,n))
    for optEquation in range(1, n + 1): # index: used exact formula as MATLAB version but subtracted 1 off of each index
        if optEquation == 1: # dx1/dt =  - x(n-1) * x(n) + x(n) * x(2) - x(1) + F
            cTrueIndex = [0, 1, 3*n-1, N-2] #index of 1, x(1), x(2)x(n), and x(n-1)x(n) respectively
            cTrueValue = [F, -1, 1, -1]
        elif optEquation == 2: # dx2/dt =  - x(n) * x(1) + x(1) * x(3) - x(2) + F
            cTrueIndex = [0, 2, n+3, 2*n]
            cTrueValue = [F, -1, 1, -1]
        elif optEquation == n: # dxn/dt = xdot(n) = - x(n-2) * x(n-1) + x(n-1) * x(1) - x(n) + F;
            cTrueIndex = [0, n, 2*n-1, N-5] #index of 1, x(n), x(1)x(n-1), and x(n-2)x(n-1) respectively
            cTrueValue = [F, -1, 1, -1]
        else:
            cTrueIndex = [0, optEquation, (optEquation-2)*(2*n-optEquation+5)//2+1, (optEquation-1)*(2*n-optEquation+4)//2+2]
            cTrueValue = [F, -1, -1, 1]
        cTrueMat[cTrueIndex[0:4], optEquation - 1] = cTrueValue[0:4]
    return cTrueMat
    
