import math
import numpy as np

def basisPursuit_Lorenz96(Vapproximate,D,optEquation,optPolynomial,opts,sigma):
    n = Vapproximate.shape[1]
    Vtest = Vapproximate[:, optEquation - 1] # optEquation between 1 to n inclusive
    soln = np.zeros((D.shape[1],1))

    # normalize dictionary and solve the optimization
    Dnormalized = np.divide(D, np.tiles(np.sqrt(np.sum(np.square(D), axis = 0)),D.shape[0],1))
    c,r,g,d = spgl1(Dnormalized, Vtest, 0, sigma, verbosity = opts.get('verbosity'), iter_lim = opts.get('iterations') )
    c = np.divide(c, np.sqrt(np.sum(np.square(D), axis = 0)) # for D normalized

    # rescale back to the monomial basis
    cRecoverIndex = np.argwhere(c)
    if optPolynomail == 'legendre':
        for i in range (0, np.len(cRecoverIndex)):
            indTmp = cRecoverIndex[i]
            if (indTmp == 0):
                soln(indTmp) = c[indTmp]
            elif (indTmp <= n):
                soln(indTmp) = c(indTmp) * math.sqrt(3)
            else:
                soln(indTmp) = c(indTmp) * 3
    else:
        soln = c
    return soln