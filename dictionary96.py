import math
import numpy as np

# Description: Construct the dictionary matrix phiX containing all multivariate monomials up to degree two for the Lorenz 96
# Input: U = [x1(t1) x2(t1) .... xn(t1)
#             x1(t2) x2(t2) .... xn(t2)
#                    ......
#             x1(tm) x2(tm) .... xn(tm)]
#        option = 'monomial' or 'legendre'
# Output: the dictionary matrix phiX of size m by N, where m = num of measurements and N = 1 + n + [n + (n-1) + ... + 1]
#         for expresion of N: 1 -> ones, n -> monomials, [n + (n-1) + ... + 1] -> combinations of monomials at degree two

def dictionary96(U, option='monomial'):
    m = U.shape[0]
    n = U.shape[1]

    # phiX num of cols = 1 + n + [n + (n-1) + ... + 1]
    phiX = np.zeros((m, (n+2)*(n+1)//2)) 
    
    # set 0-th column to ones
    phiX[:, 0] = 1

    # set the next n columns to monomials
    phiX[:,1:n+1] = math.sqrt(3)*U

    # the starting column in which we put combinations of monomials
    ind = n+1
    for k in range(0, n):
        # duplicate the kth monomial, i.e. U[:, k], n-k times
        curMono = np.tile(np.array([U[:, k]]).transpose(), (1, n-k))
        # all monomials that haven't paired with the kth monomial
        curPairs = U[:, k:n]
        # pair curMono with curPairs
        curPoly = np.multiply(curMono, curPairs)
        # assign to corresponding columns of phiX
        phiX[:, ind:ind+n-k] = 3*curPoly 
        
        # if option is legendre, then all squared terms of monomials have to be changed accordingly
        if(option == 'legendre'):
            phiX[:,ind] = math.sqrt(5)/2*(3*np.square(U[:, k])-1)

        # update ind for the next monomial column
        ind += n-k 
    
    return phiX