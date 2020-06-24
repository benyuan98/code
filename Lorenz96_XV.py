import numpy as np
from lorenz96 import lorenz96
from time_derivative import time_derivative
from scipy.integrate import odeint

# Xinit looks like x_{1,1} ... x{1, numIC}
#                          ...
#                  x_{n,1} ... x{n, numIC}

def Lorenz96_XV(F, Xinit, dt, SizeOfBurst):
    n = Xinit.shape[0]
    NumIC = Xinit.shape[1]
    Tfinal = (SizeOfBurst - 1) * dt
    tVals = [dt*i for i in range(0, SizeOfBurst)] # added this step to create time array to input into solver

    VtmpExact = np.zeros((SizeOfBurst, n))
    
    for i in range(0, NumIC):
        XinitTmp = Xinit[:, i]
        
        # solve Lorenz 96 using odeint
        sol =  odeint(func = lorenz96, y0 = XinitTmp, t = tVals, args = (n,F), tfirst = True)
        Xtmp = np.array(sol)

        # store data
        Xfull = np.copy(Xtmp) if i == 0 else np.vstack([Xfull, Xtmp])

        # Exact time-derivative by evaluating the RHS at data
        for j in range(0, SizeOfBurst):
            VtmpExact[j, :] = lorenz96(Tfinal, Xtmp[j, :], n , F)
        
        Vexact = np.copy(VtmpExact) if i == 0 else np.vstack([Vexact, VtmpExact])


        # Approximate time-derivative using finite difference
        Vtmp = time_derivative(Xtmp, dt, 2)
        Vapproximate = np.copy(Vtmp) if i == 0 else np.vstack([Vapproximate, Vtmp])

    return (Xfull, Vapproximate, Vexact)
