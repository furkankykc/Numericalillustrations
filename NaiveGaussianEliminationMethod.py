# Topic      : Naïve Gaussian Elimination Method - Roots of Equations
# Simulation : Graphical Simulation of the Method
# Language   : Python v3.8.3
# Authors    : Nathan Collier, Autar Kaw
# Translate  : Furkan Kıyıkçı
# Date       : 6 September 2002
# Abstract   : This simulation shows how the Naïve Gaussian Elimination method works for finding roots of
#              an equation f(x)=0
#

import numpy as np

np.set_printoptions(formatter={'float': "\t{: 0.2f}\t".format})

# Click the run bottom and refer to the comaand window
# These are the inputs that can be modified by the user
# n = number of equations
n = 6
# [A] = nxn coefficient matrix
A = np.array([[12., 0.0000000000007, 3., 6.0007, 5, 6],
              [1., 5., 1., 9., 7, 8],
              [13., 12., 4.0000001, 8, 4, 6],
              [5.6, 3., 7., 1.003, 7, 4],
              [1, 2, 3, 4, 5, 6],
              [6, 7, 5, 6, 7, 5]])
# [RHS] = nx1 right hand side array
RHS = np.array([22,
                7e-7,
                29.001,
                5.301,
                9,
                90])
RHS = RHS.reshape(6, 1)
print('Naïve Gaussian Elimination Method \nUniversity of South Florida \nUnited States of America \nkaw@eng.usf.edu')
print(
    '\nNOTE: This worksheet demonstrates the use of Matlab to illustrate Naïve Gaussian\nElimination, a numerical technique used in solving a system of simultaneous linear\nequations.')

print(
    '\n**************************************Introduction**************************************** \n\nOne of the most popular numerical techniques for solving simultaneous linear equations is\nNaïve Gaussian Elimination method.')
print(
    'The approach is designed to solve a set of n equations with n unknowns, \n[A][X]=[C], where [A]nxn is a square coefficient matrix, [X]nx1 is the solution vector,\nand [C]nx1 is the right hand side array.')
print('\nNaïve Gauss consists of two steps:')

print(
    '1) Forward Elimination: In this step, the coefficient matrix [A] is reduced to an\nupper triangular matrix. This way, the equations are "reduced" to one equation and\none unknown\nin each equation.')
print('2) Back Substitution: In this step, starting from the last equation, each of the unknowns\nis found.')

print('\nA simulation of Naïve Gauss Elimination Method follows.\n \n')

print('***************************************Input Data*****************************************')
print(
    'Below are the input parameters to begin the simulation. \nInput Parameters: \nn = number of equations \n[A] = nxn coefficient matrix \n[RHS] = nx1 right hand side array')

print('-----------------------------------------------------------------')
print('These are the default parameters used in the simulation. \nThey can be changed in the top part of the M-file')
print('\nn= {}'.format(n))
print(A)
print('RHS=\n', RHS)
C = np.hstack((A, RHS))
print('C=\n', C)
print(
    '--------------------------------------------------- \nWith these inputs,to conduct Naïve Gauss Elimination, Matlab will combine the [A] and\n[RHS] matrices into one augmented matrix, [C](n*(n+1)), that will facilitate the process\nof forward elimination.')
print('*************************************Forward Elimination**********************************')
print(
    'Forward elimination consists of (n-1) steps. In each step k of forward elimination,\nthe coefficient element of the kth unknown will be zeroed from every\nsubsequent equation that follows the kth row. For example, in step 2 (i.e. k=2),\nthe coefficient of x2 will be zeroed from rows 3..n.')
print(
    'With each step that is conducted, a new matrix is generated until the coefficient matrix is\ntransformed to an upper triangular matrix. Now, Matlab calculates the upper triangular\nmatrix while demonstrating the intermediate coefficient matrices that are produced for\neach step k.\n')
# Conducting k, or (n - 1) steps of forward elimination.
for k in range(n):
    # Defining the proper row elements to transform [C] into [U].

    for i in range(k + 1, n):
        # Generating    the    value    that is multiplied to    each    equation.
        multiplier = (C[i, k] / C[k, k])
        for j in range(k, n + 1):
            # Subtracting the product of the multiplier and pivot equation from the ith row to generate new rows of[U] matrix.
            C[i, j] = (C[i, j] - multiplier * C[k, j])
    print('================== Step {}'.format(k))
    print('\b =======================')
    print('C=\n', C)
    print('The elements in column #{}'.format(k))
    print('\b below C[{},{}]'.format(k, k))
print(
    'This is the end of the forward elimination steps. The coefficient matrix\nhas been reduced to an upper triangular matrix')
# --------------------------------------------------------------------------
# Creating the upper new coefficient matrix [A1]
A1 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        A1[i, j] = C[i, j]
# Now the new right hand side array, [RHS1]
RHS1 = np.zeros((n, 1))

for i in range(n):
    RHS1[i, 0] = C[i, n]
print('A1=\n', A1)
print('RHS1=\n', RHS1)

# --------------------------------------------------------------------------
print('*************************************Back substitution************************************')
print(
    '\nBack substitution begins with solving the last equation as it has only one unknown.\nThe remaining equations can be solved for using the following formula:\n')
print('x[i]=(C[i]-(sum{A[i,j]*X[j]}))/(A[i,i]')

# Defining[X] as a vector.
X = np.zeros(n)
# Solving for the nth equation as it has only one unknown.
X[-1] = RHS1[n - 1] / A1[n - 1, n - 1]
print('sonuc = ', RHS1[n - 1] / A1[n - 1, n - 1])
# Solving for the remaining(n - 1) unknowns working backwards from the (n-1)th equation to the first equation.
for i in range(n - 2, -1, -1):
    # Setting the series sum equal to zero.
    summ = 0
    for j in range(i + 1, n):
        summ += A1[i, j] * X[j]
    X[i] = (RHS1[i] - summ) / A1[i, i]
print('X=\n', X)
print('\n\nUsing back substitution, we get the unknowns as:')
X = np.rot90(X.reshape(1, n))
print('X=\n', X)
