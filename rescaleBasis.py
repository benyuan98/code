import numpy as np
import math

def rescaleBasis(c, n, p, squareDict = dict(), cubeDict = dict(), cube2VarDict = dict()):
    if p == 1:
        return rescaleLinear(c)
    elif p == 2:
        return rescaleQuadratic(c, n, squareDict)
    elif p == 3:
        return rescaleCubic(c, n, squareDict, cubeDict, cube2VarDict)
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

def rescaleQuadratic(c, n, d):
    linearCoeff = math.sqrt(3)
    quadraticCoeff = math.sqrt(5)/2
    soln = np.zeros((1,len(c))).transpose()
    cRecoverIndex = np.argwhere(c)
    
    for i in range (0, len(cRecoverIndex)):
        indTmp = cRecoverIndex[i]
        if (indTmp == 0):
            soln[indTmp] = c[indTmp]
        elif (indTmp <= n):
            soln[indTmp] = c[indTmp] * linearCoeff
        elif (indTmp in d)
            soln[indTmp] = c[indTmp] * (linearCoeff^2) * quadraticCoeff
            soln[0] = c[indTmp] * -quadraticCoeff
        else:
            soln[indTmp] = c[indTmp] * (linearCoeff^2)
    return soln


def rescaleCubic(c, n, squareDict, cubeDict, cube2VarDict):
    linearCoeff = math.sqrt(3)
    quadraticCoeff = math.sqrt(5)/2
    cubicCoeff = math.sqrt(7)/2
    soln = np.zeros((1,len(c))).transpose()
    cRecoverIndex = np.argwhere(c)
    
    for i in range (0, len(cRecoverIndex)):
        indTmp = cRecoverIndex[i]
        if (indTmp == 0):
            soln[indTmp] = c[indTmp]
        elif (indTmp <= n):
            soln[indTmp] = c[indTmp] * linearCoeff
        elif (indTmp in squareDict):
            soln[indTmp] = c[indTmp] * (linearCoeff^2) * quadraticCoeff
            soln[0] = c[indTmp] * -quadraticCoeff
        elif (indTmp in cubeDict):                  
            soln[indTmp] = c[indTmp] * 5 * cubicCoeff
            soln[cubeDict[indTmp]] = c[indTmp] * -(linearCoeff^2)  * cubicCoeff
        elif (indTmp in cube2VarDict):
            soln[indTmp] = c[indTmp] * quadraticCoeff * (linearCoeff^3)
            soln[cube2VarDict[indTmp]] = c[indTmp] * -linearCoeff * quadraticCoeff
        elif (indTmp <= (n+2)*(n+1)//2):
            soln[indTmp] = c[indTmp] * (linearCoeff^2)
        else:
            soln[indTmp] = c[indTmp] * (linearCoeff^3)
    return soln
