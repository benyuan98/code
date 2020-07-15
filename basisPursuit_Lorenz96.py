import math
import numpy as np
from spgl1 import spgl1

def basisPursuit_Lorenz96(Vapproximate,D,optEquation,optPolynomial,opts,sigma):
    n = Vapproximate.shape[1]
    Vtest = Vapproximate[:, optEquation - 1] # optEquation between 1 to n inclusive
    soln = np.zeros((D.shape[1],1))

    # normalize dictionary and solve the optimization
    factor = np.sqrt(np.sum(np.square(D), axis = 0))
    Ddivisor = np.tile(factor,(D.shape[0],1))
    Dnormalized = np.divide(D, Ddivisor)

    # Use spgl1 to solve the optimization problem min |c|_1 s.t. ||Dc - V||_2 <= sigma
    c,r,g,d = spgl1(Dnormalized, Vtest, tau = 0, sigma = sigma, verbosity = opts.get('verbosity'), iter_lim = opts.get('iterations'))
    c = np.divide(c, factor) # for D normalized

    # rescale back to the monomial basis
    cRecoverIndex = np.argwhere(c)
    if optPolynomial == 'legendre':
        for i in range (0, len(cRecoverIndex)):
            indTmp = cRecoverIndex[i]
            if (indTmp == 0):
                soln[indTmp] = c[indTmp]
            elif (indTmp <= n):
                soln[indTmp] = c[indTmp] * math.sqrt(3)
            else:
                soln[indTmp] = c[indTmp] * 3
    else:
        for i in range (0, len(cRecoverIndex)):
            indTmp = cRecoverIndex[i]
            soln[indTmp] = c[indTmp] 
    return soln