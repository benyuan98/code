import numpy as np
import math

def buildDependencyDictionary(U, p, DepMat):
    if p == 1:
        return linearDict(U)
    elif p == 2:
        return quadraticDepDict(U, DepMat)
    elif p == 3:
        return cubicDepDict(U, DepMat)
    return None

def makeCol(row): # input is a row, output as np array column
    return np.array([row]).transpose()

def linearDict(U):
    m,n = U.shape
    phiX = np.zeros((m, n+1))
    phiX[:, 0] = 1
    phiX[:, 1:n+1] = linearLegendre(U)
    return phiX
def linearLegendre(U):
    return math.sqrt(3) * U

def quadraticDepDict(U, DepMat):
    n = U.shape[1]

    # set the first n + 1 columns to linear dictionary
    phiX = linearDict(U)

    # stores the indices of squared monomials in phiX
    squaredMonomial = dict()

    # for (x_a)(x_b) at index i, variablePairs[i] = (a,b)
    variablePairs = dict()
    
    currInd = n + 1
    for i in range (n):
        colCounter = 0

        # check for pairs on dependency matrix
        for j in range(i,n):
            if DepMat[i,j] == 1:
                currCol = makeCol(linearLegendre(U[:,j]))
                if colCounter == 0: # on the diagonal, combine with self
                    curPairs = currCol
                    squaredMonomial[currInd] = i    
                else:
                    curPairs = np.append(curPairs, currCol, 1)
                
                variablePairs[currInd + colCounter] = (i, j)

                colCounter += 1
        
        curMono = linearLegendre(np.tile(makeCol(U[:, i]), (1, colCounter)))
        curQuad = np.multiply(curMono, curPairs)
        
        # append to phiX
        phiX = np.append(phiX, curQuad,1)
        
        # since is legendre, squared terms of monomials have to be changed accordingly
        phiX[:,currInd] = squaredLegendre(U[:,i])

        currInd += colCounter

    return (phiX, squaredMonomial, variablePairs)

def squaredLegendre(col):
    return math.sqrt(5)/2*(3*np.square(col)-1)

def cubicDepDict(U, DepMat):
    n = U.shape[1]

    # set the first columns to quadratic dictionary
    phiX, squaredMonomial, variablePairs = quadraticDepDict(U, DepMat)

    # stores the indices of cubed monomials in phiX
    cubicMonomial = dict()

    # for (x_a)(x_b)^2 at index i, cubicTwoVariables[i] = a
    cubicTwoVariables = dict()
    
    totalQuadCol = phiX.shape[1]
    currInd = totalQuadCol

    # loop through quadratic terms
    for i in range(n+1, totalQuadCol):
        colCounter = 0
        (a,b) = variablePairs[i]
        if a == b: # for squared terms
            for j in range(a, n):
                if DepMat[a,j] == 1:
                    currCol = makeCol(linearLegendre(U[:,j]))
                    if colCounter == 0: # on the diagonal, and note a == b, so first term is a cubed term 
                        curMono = currCol
                        cubicMonomial[currInd] = a
                    else:
                        curMono = np.append(curMono, currCol, 1)
                    colCounter += 1
        else:
            sumDepRow = np.add(DepMat[a], DepMat[b])
            for j in range(b, n):
                if sumDepRow[j] == 2:
                    currCol = makeCol(linearLegendre(U[:,j]))
                    if colCounter == 0: # on the diagonal, first term created is (x_a)(x_b)^2
                        curMono = currCol
                        cubicTwoVariables[currInd] = a 
                    else:
                        curMono =  np.append(curMono, currCol, 1)
                    colCounter += 1
        curQuad = linearLegendre(np.tile(makeCol(U[:, i]), (1, colCounter)))
        curCube = np.multiply(curQuad, curMono)

        # append to phiX
        phiX = np.append(phiX, curCube,1)
        
        # since is legendre, cubed terms have to be changed accordingly
        if a == b:
            phiX[:,currInd] = cubicLegendre(U[:,a])

        currInd += colCounter

    return (phiX, squaredMonomial, cubicMonomial, cubicTwoVariables)

def cubicLegendre(col):
    return math.sqrt(7)/2*(5*np.power(col, 3)-3*col)
