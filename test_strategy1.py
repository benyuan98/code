import math
import random
import numpy as np
import Lorenz96_true_coefficients as coeff
from scipy.sparse import csr_matrix

## ODE parameters
n = 50              # number of variables
F = 8.0             # constant of Lorenz 96
optEquation = 10    #Equation to test
NumIC = 100         #number of initial conditions

# Other parameters

N = (n+1)*(n+2)/2   # number of columns of the dictionary matrix 
dt = 0.001          # time step
SizeOfBurst = 5     # size of each burst i.e. number of measurements from a given trajectory
epsilon = 0.5       # used in thm 3.1
cStar = 1           # universal constant in thm 3.1

lowerBoundNumIC = round(5*math.log(N)*math.log(1/epsilon)) # s*log(N)*log(1/varepsilon)
upperBoundNumIC = round(N/SizeOfBurst)

print("The number of initializations NumIC should be at least " + str(lowerBoundNumIC)
+ "c and be smaller than " + str(upperBoundNumIC))

# spgl1 equivalent parameters
opts = {'verbosity': 0, 'iterations': 1000} # in basis pursuit, use opts.get('verbosity')

optPolybomial = 'legendre'  #'legendre or 'monomial'

cTrueMat = coeff.Lorenz96_true_coefficients(n,F)

## Data generated from K bursts starting from K random initializations 
Xint = 2*np.random.rand(n,NumIC)-1 # initialization is a unifeorm random variable on [-1,1]
Xfull, Vapproximate,Vexact = Lorenz96_XV(F,Xint,dt,SizeOfBurst)

# Built dictionary
D = dictionary96(Xfull,optPolynomial)

## Basis Pursuit Denoising Problem
sigma = 2 * np.linalg.norm(Vapproximate[:,optEquation - 1]-Vexact[:,optEquation - 1],2)
soln = basisPursuit_Lorenz96(Vapproximate,D,optEquation,optPolynomial,opts,sigma)

## Print out
print('The nonzero terms of the true coefficients from Equation '+ str(optEquation))
csr_matrix(c_true_mat[:,optEquation - 1]).eliminate_zeros
print(['The nonzero terms of the recovered coefficients from Equation '+ str(optEquation)])
csr_matrix(soln)