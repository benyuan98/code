import numpy as np
import pandas as pd
import math
import pickle

def listToTxt(values, fileName):
    with open('fileName', 'wb') as output:
        pickle.dump(values, output)

def txtToList(fileName):
    with open ('fileName', 'rb') as fp:
        values = pickle.load(fp)
    return values

# encode the form of polynomials as follows:
# 0: ones, 1: monomials, 2: Xi^2, 3: Xi*Xj
# 4: Xi^3, 5: Xi^2*Xj, 6: Xi*Xj^2, 7: Xi*Xj*Xk
# i < j < k
def makeLegendre(col1, col2, form):
    # col1: Xi, col2: Xi
    if(form == 2):
        coeff = math.sqrt(5)/2
        return coeff * (3*np.multiply(col1, col2) - 1)

    # col1: Xi, col2: Xj
    elif(form == 3):
        return 3*np.multiply(col1, col2)

    # col1: Xi, col2: Xi^2
    elif(form == 4):
        coeff = math.sqrt(7)/2
        return coeff*(5*np.multiply(col1, col2)-3*col1)
    
    # col1: Xi, col2: Xi*Xj
    elif(form == 5):
        coeff = math.sqrt(5)*math.sqrt(3)/2
        Xj = np.divide(col2, col1)
        Xi2 = np.multiply(col1, col1)
        return coeff * np.multiply(3*Xi2-1, Xj)
    
    # col1: Xi, col2: Xj^2
    elif(form == 6):
        coeff = math.sqrt(5)*math.sqrt(3)/2
        return coeff * (np.multiply(3*col2-1), col1)
    
    # col1: Xi, col2: Xj*Xk
    elif(form == 7):
        return 3*math.sqrt(3)*np.multiply(col1, col2)

    

def buildDictionary(U, A, p):
    if p == 1:
        return linearDict(U)
    elif p == 2:
        return quadraticDict(U, A, Legendre=True)
    elif p == 3:
        return cubicDict(U, A, Legendre=True)
    return None

def linearDict(U):
    m,n = U.shape
    phiX = np.zeros((m, n+1))
    phiX[:, 0] = 1
    phiX[:, 1:n+1] = U
    return phiX


# U: input data
# A: Dependency Matrix
def quadraticDict(U, A, Legendre=True):
    m,n = U.shape
    assert(n == A.shape[1])

    phiXcols = 0
    # column names of phiX
    allVarNames = ['1']
    formInfo = [0]

    for i in range(n):
        phiXcols += int(np.sum(A[i, i:]))
        allVarNames.append('X_'+str(i+1))
        formInfo.append(1)
    
    # 1: ones, n: monomials, phiXcols: quadratic
    phiX = np.zeros((m, 1 + n + phiXcols)) 
    
    # set 0-th column to ones 
    phiX[:, 0] = 1

    # set the next n columns to monomials
    if(Legendre):
        phiX[:,1:n+1] = math.sqrt(3)*U
    else:
        phiX[:,1:n+1] = U

    # set phiXind to be the starting index for storing quadratic terms
    phiXind = n+1

    # looping through all variables
    for i in range(n):
        # curCols: how many other variables X_{i+1} needs to pair
        curCols = int(np.sum(A[i, i:]))
        curPart = np.zeros((m, curCols))
        curInd = 0
        # column names of curPart
        varNames = []
        # column variables information (to be exported as txt)
        varInfo = []
        for j in range(i, n):
            if(A[i, j] == 1):
                # record the form of the quadratic
                if(i == j):
                    formInfo.append(2)
                else:
                    formInfo.append(3)

                # record column names
                varNames.append('X'+str(i+1)+'X'+str(j+1))
                varInfo.append(i+1)

                if(Legendre):
                    prod = makeLegendre(U[:, i], U[:, j], formInfo[-1])
                else:
                    prod = np.multiply(U[:, i], U[:, j])

                curPart[:, curInd] = prod
                phiX[: phiXind] = prod
                
                curInd += 1
                phiXind += 1
        
        allVarNames += varNames
        # export curPart as csv file below
        csvFileName = 'X' + str(i+1) + 'Quadratic'
        # make carPart into a pandas dataframe and save it to a csv file
        curPart_df = pd.DataFrame(curPart)
        curPart_df.columns = varNames
        curPart_df.to_csv(csvFileName)
        # record column variables to a txt file
        txtFileName = 'X' + str(i+1) + 'Quadratic' + 'Info.txt'
        listToTxt(varInfo, txtFileName)
        
    listToTxt(formInfo, 'formInfo.txt')
    phiX = pd.DataFrame(phiX)
    phiX.columns = allVarNames
    return phiX 


    
# U: input data
# A: Dependency Matrix
def cubicDict(U, A, Legendre=True):
    m,n = U.shape
    
    phiX = quadraticDict(U, A, Legendre)
    phiXcolNames = phiX.columns
    phiX = phiX.to_numpy()
    # record all cubic variable names
    cubicVarNames = []

    formInfo = txtToList('formInfo.txt')

    for i in range(n):

        for j in range(i, n):
            if(A[i, j] == 1):
                curPart = pd.read_csv('X_' + str(j+1) + 'Quadratic')
                curPart = curPart.to_numpy()
                curPartCols = txtToList('X' + str(j+1) + 'Quadratic' + 'Info.txt')

                for k in range(len(curPartCols)):
                    curColVar = curPartCols[k]
                    if(A[i, curColVar-1] == 1):
                        
                        # add an extra column of zeros to phiX
                        phiX = np.append(phiX, np.zeros((m, 1)), axis = 1)
                        # record column name
                        cubicVarNames.append('X'+str(i+1)+'X'+str(j+1)+'X'+str(curColVar))
                        if(i == j and i == curColVar):
                            formInfo.append(4)
                        elif(i == j and i != curColVar):
                            formInfo.append(5)
                        elif(i != j and j == curColVar):
                            formInfo.append(6)
                        elif(i != j and j !=  curColVar):
                            formInfo.append(7)
                        
                        if(Legendre):
                            prod = makeLegendre(U[:, i], curPart[:, k], formInfo[-1])
                        else:
                            prod = np.multiply(U[:, i], curPart[:, k])
                        
                        # set the exta column to be prod
                        phiX[:, -1] = prod

    phiXcolNames += cubicVarNames
    phiX = pd.DataFrame(phiX)
    phiX.columns = phiXcolNames

    return phiX