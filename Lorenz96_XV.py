import numpy as np
from lorenz96 import lorenz96
from time_derivative import time_derivative
from scipy.integrate import solve_ivp
#from scipy.integrate import odeint

def Lorenz96_XV(F, Xint, dt, SizeOfBurst):
    n = Xint.shape[0]
    NumIC = Xint.shape[1]
    Tfinal = (SizeOfBurst - 1) * dt
    tVals = [dt*i for i in range(0, SizeOfBurst)] # added this step to create time array to input into solver

    VtmpExact = np.zeros((SizeOfBurst, n))


    for i in range(0, NumIC):
        XintTmp = Xint[:, i]
        
        # solve Lorenz 96 using odeint
        sol =  odeint(func = lorenz96, y0 = XintTmp, t = tVals, args = (n,F), tfirst = True)
        Xtmp = np.array(sol)
        
        #solve Lorenz 96 using solve_ivp
        #sol = solve_ivp(fun = lorenz96, t_span = [0, Tfinal], y0 = XintTmp, t_eval = tVals, args = (n,F))
        #Xtmp = np.array(sol.y).transpose()

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
