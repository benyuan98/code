import math
import random
import numpy as np
from Lorenz96_true_coefficients import Lorenz96_true_coefficients
from Lorenz96_XV import Lorenz96_XV
from dictionary96 import dictionary96
from basisPursuit_Lorenz96 import basisPursuit_Lorenz96

## ODE parameters
n = 50              # number of variables
F = 8.0             # constant of Lorenz 96
optEquation = 10    # Equation to test
NumIC = 100       # number of initial conditions

# Other parameters

N = (n+1)*(n+2)/2   # number of columns of the dictionary matrix 
dt = 0.001         # time step
SizeOfBurst = 5   # size of each burst i.e. number of measurements from a given trajectory
epsilon = 0.5       # used in thm 3.1
cStar = 1           # universal constant in thm 3.1

lowerBoundNumIC = round(5*math.log(N)*math.log(1/epsilon)) # s*log(N)*log(1/varepsilon)
upperBoundNumIC = round(N/SizeOfBurst)

print("The number of initializations NumIC should be at least " + str(lowerBoundNumIC)
+ "c and be smaller than " + str(upperBoundNumIC))

# spgl1 equivalent parameters
opts = {'verbosity': 0, 'iterations': 1000} 

optPolynomial = 'legendre'  #'legendre or 'monomial'

cTrueMat = Lorenz96_true_coefficients(n,F)

## Data generated from K bursts starting from K random initializations 
Xinit = 2*np.random.rand(n, NumIC)-1 # initialization is a uniform random variable on [-1,1]
Xfull, Vapproximate, Vexact = Lorenz96_XV(F,Xinit,dt,SizeOfBurst)

# Built dictionary
D = dictionary96(Xfull,optPolynomial)

## Basis Pursuit Denoising Problem
sigma = 2*np.linalg.norm(Vapproximate[:,optEquation - 1]-Vexact[:,optEquation - 1])
soln = basisPursuit_Lorenz96(Vapproximate, D, optEquation, optPolynomial, opts, sigma)

## Print out
print('The nonzero terms of the true coefficients from Equation '+ str(optEquation))
A = np.array(cTrueMat[:,optEquation - 1])
print(np.argwhere(A != 0))
print(A[A!=0])
print(['The nonzero terms of the recovered coefficients from Equation '+ str(optEquation)])
print(np.argwhere(soln != 0))
print(soln[soln!=0])