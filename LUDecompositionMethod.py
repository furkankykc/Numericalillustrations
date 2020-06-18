# Topic      : LU Decomposition method
# Language   : Python v3.8.3
# Authors    : Nathan Collier, Autar Kaw
# Translate  : Furkan Kıyıkçı
# Date       : 6 September 2002

# Click the run bottom and refer to the comand window
# These are the inputs that can be modified by the user
import numpy as np

np.set_printoptions(formatter={'float': "\t{: 0.2f}\t".format})
# n = number of equations
n = 6
# [A] = nxn coefficient matrix
A = np.array([[12., 0.1234567890987654, 3., 6.7, 5, 6],
              [1., 5.053e9, 1., 9., 7, 8],
              [13., 12., 4.0000001, 8, 4, 6],
              [5.6, 3., 7., 1.003, 7, 4],
              [1, 2, 3, 4, 5, 6],
              [6, 7, 5, 6, 7, 5]])
RHS = np.array([22, 7e-7, 29.001, 5.301, 9, 90])

print('LU Decomposition Method')
print('University of South Florida')
print('United States of America')
print('kaw@eng.usf.edu')

print(
    'NOTE: This worksheet demonstrates the use of Matlab to illustrate LU Decomposition method,\na technique used in solving a system of simultaneous linear equations.')
# --------------------------------------------------------------------------
print('  ')
print('**************************************Introduction***************************************')
print(' ')
print(
    'When solving multiple sets of simultaneous linear equations with the same coefficient\nmatrix but different right hand sides,')
print(
    '\bLU Decomposition is advantageous over other\nnumerical methods in that it proves to be numerically more efficient in computational\ntime ')
print(
    '\bthan other techniques.\nIn this worksheet, the reader can choose a system of equations and see how each\nstep of LU decomposition method is conducted.')

print('\n\nLU Decomposition method is used to solve a set of simultaneous linear equations,\n[A] [X] = [C],')
print('\b where [A]nxn is a non-singular square coefficient matrix, [X]nx1 is the\nsolution vector,')
print('\band [C]nx1 is the right hand side array.\nWhen conducting LU decomposition method,')
print('\bone must first decompose the coefficient matrix\n[A]nxn into a lower triangular matrix [L]nxn,')
print(
    '\b and upper triangular matrix [U]nxn.\nThese two matrices can then be used to solve for the solution vector [X]nx1\nin the following sequence:')
print('Recall that')
print('[A] [X] = [C].')
print('Knowing that')
print('[A] = [L] [U]')
print('then first solving with forward substitution')
print('[L] [Z] = [C]')
print('and then solving with back substitution')
print('[U] [X] = [Z]')
print('gives the solution vector [X].')
# -------------------------------------------------------------------------

print('**************************************Input Data***********************************')
print('Below are the input parameters to begin the simulation.')
print('Input Parameters:')
print('n = number of equations')
print('[A] = nxn coefficient matrix')
print('[RHS] = nx1 right hand side array')
print('n=%d'.format(n))

print('A=\n', A)
print('RHS=\n', RHS)
print('***********************************************************************************')
print('************************** LU Decomposition Method ********************************')
print('***********************************************************************************')
print('\nThe following sections divide LU Decomposition method into 3 steps:\n')
print('1.) Finding the LU decomposition of the coefficient matrix [A]nxn')
print('2.) Forward substitution')
print('3.) Back substitution')
print(' ')

# --------------------------------------------------------------------------
# LU Decomposition
print('-------------------------------Finding the LU Decomposition-------------------------')
print(' ')
print('How does one decompose a non-singular matrix [A], that is how do you find [L] and [U]?')
print(
    'This worksheet decomposes the coefficient matrix [A] into a lower triangular matrix [L]\nand upper triangular matrix [U], given [A] = [L][U].')
print(
    '\f For [U], the elements of the matrix are exactly the same as the coefficient matrix one\nobtains at the end of forward elimination steps in Naïve Gauss Elimination.')
print(
    '\f For [L], the matrix has 1 in its diagonal entries. The non-zero elements are\nmultipliers that made the corresponding elements zero in the upper triangular matrix\nduring forward elimination.')

L = np.zeros((n, n))
U = np.zeros((n, n))
# Initializing diagonal of [L] to be unity.
for i in range(n):
    L[i, i] = 1.0

# Dumping [A] matrix into a local matrix [AA]
AA = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        AA[i, j] = A[i, j]

# Conducting forward elimination steps of Naïve Gaussian Elimination to obtain [L] and [U] matrices.

for k in range(n):
    for i in range(k + 1, n):
        # Computing multiplier values.
        multiplier = AA[i, k] / AA[k, k]
        print(i, k, '=', multiplier)
        # Putting multiplier in proper row and column of [L] matrix.
        L[i, k] = multiplier
        # print(L, i, k, multiplier)
        for j in range(k, n):
            # Eliminating(i - 1) unknowns from the ith row to generate an upper triangular matrix.
            AA[i, j] = AA[i, j] - multiplier * AA[k, j]
    # Dumping the end of forward elimination coefficient matrix into the [U] matrix.

for i in range(n):
    for j in range(i, n):
        U[i, j] = AA[i, j]

print('L=\n', L)
print('U=\n', U)
# --------------------------------------------------------------------------
# Forward Substitution
print('\n\n------------------------------- Forward Substitution---------------------------------\n')
print(
    'Now that the [L] and [U] matrices have been formed, the forward substitution step,\n[L] [Z] = [C], can be conducted, beginning with the first equation as it has only one\nunknown,\n')
print('z[1] = c[1]/l[1, 1]')
print('\nSubsequent steps of forward substitution can be represented by the following formula:\n')
print('z[i] = (c[i]-(Sum(l[i, j]*z[j], j = 1 .. i-1))[i = 2 .. n])/l[i, i]')

# Defining the [Z] vector.
Z = np.zeros(n)
# Solving for the first equation as it has only one unknown.
Z[0] = RHS[0] / L[0, 0]
# Solving for the remaining (n-1) unknowns
for i in range(1, n):
    sum = 0
    # Generating the summation term
    for j in range(i):
        sum = sum + L[i, j] * Z[j]

    Z[i] = (RHS[i] - sum) / L[i, i]
Z = np.transpose(Z)

print('Z=\n', Z)
# --------------------------------------------------------------------------
# Back Substitution
print('\n-----------------------------------Back Substitution------------------------------------\n')
print(
    'Now that [Z] has been calculated, it can be used in the back substitution step,\n[U] [X] = [Z], to solve for solution vector [X]nx1, where [U]nxn is the upper triangular\nmatrix calculated in Step 2.1, and [Z]nx1 is the right hand side array.')
print('Back substitution begins with the nth equation as it has only one unknown:\n')
print('xn = zn/U(n, n)')
print('\nThe remaining unknowns are solved for using the following formula:\n')
print('xi = (zi-(Sum(U[i, j]*X[j], j = i+1 .. n))[i = n-1 .. 1])/U[i, i]')
# Defining the [X] vector.
X = np.zeros(n)
# Solving for the nth equation as it has only one unknown.

X[n - 1] = Z[n - 1] / U[n - 1, n - 1]
print(Z[n - 1] / U[n - 1, n - 1])
# Solving for the remaining (n-1) unknowns working backwards from the (n-1)th equation to the first equation.
for i in range(n-2, -1, -1):
    # Initializing series sum to zero.
    sum = 0
    # Calculating summation term
    for j in range(i + 1, n):
        sum += U[i, j] * X[j]
    # Calculating solution vector [X].
    X[i] = (Z[i] - sum) / U[i, i]
X = np.transpose(X)
print('X=\n', X)
# ------------------------------------------------------------------------
