import numpy as np
import math

def rescaleBasis(c, p, d = dict()):
    if p == 1:
        return rescaleLinear(c)
    elif p == 2:
        return rescaleQuadratic(c, d)
    elif p == 3:
        return rescaleCubic(c, d)
    return None

def rescaleLinear(c):
    linearCoeff = 1
    soln = np.zeros((1,len(c))).transpose()

    cRecoverIndex = np.argwhere(c)
    for i in range (0, len(cRecoverIndex)):
        indTmp = cRecoverIndex[i]
        if (indTmp == 0):
            soln[indTmp] = c[indTmp]
        else:
            soln[indTmp] = c[indTmp] * linearCoeff
    return soln

def rescaleQuadratic(c, d):
    soln = np.zeros((1,len(c))).transpose()
    cRecoverIndex = np.argwhere(c)
    
    for i in range (0, len(cRecoverIndex)):
        indTmp = cRecoverIndex[i]
        if (indTmp == 0):
            soln[indTmp] = c[indTmp]
        elif (indTmp <= n):
            soln[indTmp] = c[indTmp] * math.sqrt(3)
        elif (indTmp in d)
            soln[indTmp] = c[indTmp] * 3 * math.sqrt(5)/2
            soln[0] = c[indTmp] * -math.sqrt(5)/2
        else:
            soln[indTmp] = c[indTmp] * 3
    return soln


def rescaleCubic(c, d):
    cubicCoefficient = 1
    soln = np.zeros((1,len(c))).transpose()
    cRecoverIndex = np.argwhere(c)
    
    for i in range (0, len(cRecoverIndex)):
        indTmp = cRecoverIndex[i]
        if (indTmp == 0):
            soln[indTmp] = c[indTmp]
        elif (indTmp <= n):
            soln[indTmp] = c[indTmp] * math.sqrt(3) # maybe change constant
        elif (indTmp in d):
            if d[indTmp] == -1: # squared term
                soln[indTmp] = c[indTmp] * 3 * math.sqrt(5)/2
                soln[0] = c[indTmp] * -math.sqrt(5)/2
            else:                   # cubed term
                soln[indTmp] = c[indTmp] * 5 / 2 * cubicCoefficent
                soln[d[indTmp]] = c[indTmp] * -3 / 2 * cubicCoefficient
        elif (indTmp <= (n+2)*(n+1)//2):
            soln[indTmp] = c[indTmp] * 3
        else:
            soln[indTmp] = c[indTmp] * cubicCoefficient
    return soln