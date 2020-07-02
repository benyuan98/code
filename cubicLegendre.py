import numpy as np
import math

def buildDictionary(U, p):
    if p == 1:
        return linearDict(U)
    elif p == 2:
        return quadraticDict(U)
    elif p == 3:
        return cubicDict(U)
    return None

def linearDict(U):
    m,n = U.shape
    phiX = np.zeros((m, n+1))
    phiX[:, 0] = 1
    phiX[:, 1:n+1] = linearLegendre(U)
    return phiX
def linearLegendre(U):
    return math.sqrt(3) * U

def quadraticDict(U):
    m,n = U.shape

    # phiX num of cols = 1 + n + [n + (n-1) + ... + 1]
    phiX = np.zeros((m, (n+2)*(n+1)//2)) 
    
    # set the first n + 1 columns to linear dictionary
    phiX[:,0:n+1] = linearDict(U)
    
    # stores the indices of squared monomials in phiX
    squaredMonomial = dict()

    # the starting column in which we put combinations of monomials
    ind = n+1
    for k in range(0, n):
        # duplicate the kth monomial, i.e. U[:, k], n-k times
        curMono = np.tile(np.array([U[:, k]]).transpose(), (1, n-k))
        # all monomials that haven't paired with the kth monomial
        curPairs = U[:, k:n]
        # pair curMono with curPairs
        curQuad = np.multiply(curMono, curPairs)
        # assign to corresponding columns of phiX
        phiX[:, ind:ind+n-k] = curQuad
        
        # since is legendre, then all squared terms of monomials have to be changed accordingly
        phiX[:,ind] = squaredLegendre(U[:,k])
        squaredMonomial[ind] = k

        # update ind for the next monomial column
        ind += n-k 
    
    return (phiX, squaredMonomial)
def squaredLegendre(col):
    return math.sqrt(5)/2*(3*np.square(col)-1)

def cubicDict(U):
    m,n = U.shape
    
    phiX = np.zeros((m, 1 + n + n*(n+1)//2 + n^2 + n*(n-1)*(n-2)//6)) 
    quadD, squareindex = quadraticDict(U)

    # index in phiX 
    cubicInd = 1 + n + n*(n+1)//2
    # index in quadD
    quadInd = n + 1

    # set the first 1 + n + n*(n+1)/2 columns to be quadD
    phiX[:,:quadD.shape[1]] = quadD

    # stores the indices of cubed monomials in phiX
    cubicMonomial = dict()
    
    # stores indices of terms with squared monomial times a different monomial
    cubicTwoVariables = dict()

    for i in range(n):
        # number of cols X_{i+1} need to duplicate
        dim = (n+1-i)*(n-i)//2
        curMono = np.tile(np.array([U[:, i]]).transpose(), (1, dim))

        curCube = np.multiply(curMono, quadD[:,quadInd:])
        phiX[:,cubicInd:cubicInd+dim] = curCube
        
        # since is legendre, all cubed terms of monomials have to be changed accordingly
        phiX[:, cubicInd] = cubicLegendre(U[:,i])
        
        cubicMonomial[cubicInd] = i+1
        
        # add indices of terms with a squared monomial times a different monomial
        for j in range(n-i-1):
            cubicTwoVariables[cubicInd+j] = i+j+2
        for k in range(n-i, 1, -1):
            cubicTwoVariables[cubicInd+k] = i
        
        cubicInd += dim
        quadInd += (n-i)

    return (phiX, cubicMonomial)

def cubicLegendre(col):
    return math.sqrt(7)/2*(5*np.power(col, 3)-3*col)


def nChoose3R1(n):
    numerator = (n+2)*(n+1)*n
    denominator = 6
    return (numerator//denominator)

def nChoose3R2(n):
    result = 0
    for i in range(n, 0, -1):
        result += i*(i+1)//2
    return result

#print(nChoose3R1(100))
#print(nChoose3R2(100))
