# Language: Matlab 7.4.0 (R2007a)
# Revised:
#   July 7, 2008

# Authors:
#   Nathan Collier, Luke Snyder, Autar Kaw
#   University of South Florida
#   kaw@eng.usf.edu
#   Website: http://numericalmethods.eng.usf.edu
# Translate  : Furkan Kıyıkçı
# Purpose
#   To illustrate the Lagrangian method of interpolation using Matlab.
# Keywords
#   Interpolation
#   Lagrange method of interpolation
# Clearing all data, variable names, and files from any other source and
# clearing the command window after each sucesive run of the program.
# Inputs:
#    This is the only place in the program where the user makes the changes
#    based on their wishes.
from math import log10

import numpy as np
from sympy import *
import matplotlib.pyplot as plt

# Enter arrays of x and y data and the value of x at which y is desired.
#  Array of x-data
x = np.array([10, 0, 20, 15, 30, 22.5])
#  Array of y-data
y = np.array([227.04, 0, 517.35, 362.78, 901.67, 602.97])
# Value of x at which y is desired
xdesired = 16
#  *************************************************************************
print('\n\nLagrange Method of Interpolation')
print('\nUniversity of South Florida')
print('United States of America')
print('kaw@eng.usf.edu')
print('\nNOTE: This worksheet utilizes Matlab to illustrate the concepts')
print('of the Lagrange method of interpolation.')

print('\n**************************Introduction***************************')

print('\nThe following simulation uses Matlab to illustrate the Lagrange method')
print('of interpolation.  Given n data points of y vs x,  it is required to')
print('find the value of y at a particular value of x using first, second, or')
print('third order interpolation.  It is necessary to first pick the needed data')
print('points and use those specific points to interpolate the data.  This')
print('method differs from the Direct Method of interpolation and the Newton')
print('Divided Difference Method with the use of a "weight" function to ')
print('approximate the value of y at the x values that bracket the "xdesired"')
print('term.  The result is then the summation of these weight functions multiplied')
print('by the known value of y at the corresponding value of x.')

print('\n\n***************************Input Data******************************')
print('\n')
print('x array of data:')
print('x=\n', x)
print('y array of data:')
print('y=\n', y)
print('The value of x at which y is desired, xdesired = #g', xdesired)

print('\n***************************Simulation******************************')
# Determining whether or not "xdesired" is a valid point to ask for given
# the range of the x data.
high = max(x)
low = min(x)
if xdesired > high:
    print('\nThe value entered for "xdesired" is too high. Please pick a smaller value')
    print('that falls within the range of x data.')
    exit()
elif xdesired < low:
    print('\nThe value entered for "xdesired" is too low.  Please pick a larger value')
    print('that falls within the range of x data.')
    exit()
else:
    print('\nThe value for "xdesired" falls within the given range of xdata.  The')
    print('simulation will now commence.')
# The following considers the x and y data and selects the two closest points to xdesired
# that also bracket it
n = np.size(x)
comp = np.absolute(x - xdesired)
c = np.min(comp)
ci = 0
for i in range(n):
    if comp[i] == c:
        ci = i
# The following sequence of if statements determines if the value examined
# in the x array is greater than or less than the value "xdesired".  Once
# this is determined, the remaining lines find the necessary points around
# the "xdesired" variable.
ne = np.zeros(n)
if x[ci] < xdesired:
    q = 1
    for i in range(n):
        if x[i] > xdesired:
            ne[q] = x[i]
            q += 1
    b = min(ne)
    for i in range(n):
        if x[i] == b:
            bi = i
if x[ci] > xdesired:
    q = 1
    for i in range(n):
        if x[i] < xdesired:
            ne[q] = x[i]
            q += 1
    b = max(ne)

    for i in range(n):
        if x(i) == b:
            bi = i
# If more than two values are needed, the following selects the subsequent values and puts
# them into a matrix, preserving the orginal data order.
A = np.zeros((n, n))
print(A.shape)
for i in range(n):
    A[i, 2] = i
    A[i, 1] = comp[i]

A = np.argsort(A)
for i in range(n):
    A[i, 3] = i

A = np.argsort(A)
d = A[1:n, 3]
if d[bi] != 2:
    temp = d[bi]
    d[bi] = 1
    for i in range(n - 1):
        if i != bi and i != ci and d[i] <= temp:
            d[i] = d[i] + 1

    d[ci] = 1
######### LINEAR INTERPOLATION #########

# Pick two data points
datapoints = 2
p = 0
xdata = np.zeros(datapoints)
ydata = np.zeros(datapoints)
for i in range(n - 1):
    if p >= datapoints:
        break
    if d[i] <= datapoints:
        xdata[p] = x[i]
        ydata[p] = y[i]
        p += 1
# Setting up the Lagrangian polynomial
z = symbols('z')
L0 = (z - xdata[1]) / (xdata[0] - xdata[1])
L01 = L0.subs(z, xdesired)
L1 = (z - xdata[0]) / (xdata[1] - xdata[0])
L11 = L1.subs(z, xdesired)
fl = L0 * ydata[0] + L1 * ydata[1]
fxdesired = fl.subs(z, xdesired)
fprev = fxdesired

# displaying the outputs:
print('\nLINEAR INTERPOLATION:')
print('\nx data chosen: x1 = {}, x2 = {}', xdata[0], xdata[1])
print('\ny data chosen: y1 = {}, y2 = {}', ydata[0], ydata[1])

# displaying the values for the Lagrangian polynomial.
print('\nLagrange weight function values, L0 = {}, L1 = {}'.format(L01, L11))

# Using concactenation to properly printlay the final function inside the
# command window.
print('\n')
str1 = 'Final function, f(x) = ', str(L01), '(', str(ydata[1]), ')'
if L11 > 0:
    str2 = ' + ', str(L11), '(', str(ydata[1]), ')'
else:
    str2 = str(L11), '(', str(ydata[1]), ')'

# Putting all the strings together.
finalstr = str1 + str2
print('final = ', finalstr)

# displaying the final value of f(x) at xdesired.
print('\nThe value of f(x) at xdesired, f({}) = {}'.format(xdesired, fxdesired))

# Plotting the results:
z1 = np.arange(np.min(xdata), np.max(xdata), 0.1)
fz = [fl.subs(z, zz) for zz in z1]

#  Creating the first plot for Linear Lagrange Interpolation.
fig, ax = plt.subplots(2, figsize=(20, 10))  # Create a figure containing a single axes.

ax[0].plot(z1, fz)
ax[0].set_title('Linear Interpolation (Data Points Used)')
ax[0].set_ylabel('f(x)')
ax[0].set_xlabel('x data')
ax[0].plot(xdata, ydata, 'ro')
ax[0].plot(xdesired, fxdesired, 'kx')
# ax[1] = plt.subplot(211)
ax[1].plot(z1, fz)
ax[1].set_title('Linear Interpolation (Full Data Set)')
ax[1].set_ylabel('f(x)')
ax[1].set_xlabel('x data')
ax[1].plot(x, y, 'ro')
ax[1].plot(xdesired, fxdesired, 'kx')
ax[1].set_xlim([np.min(x), np.max(x)])
ax[1].set_ylim([np.min(y), np.max(y)])
fig.show()

######### QUADRATIC INTERPOLATION #########

# Pick three data points
datapoints = 3
p = 0
xdata = np.zeros(datapoints)
ydata = np.zeros(datapoints)
for i in range(n - 1):
    if p >= datapoints:
        break
    if d[i] <= datapoints:
        xdata[p] = x[i]
        ydata[p] = y[i]
        p += 1

# Setting up the Lagrangian polynomial
z = symbols('z')
L0 = ((z - xdata[1]) * (z - xdata[2])) / ((xdata[0] - xdata[1]) * (xdata[0] - xdata[2]))
L01 = L0.subs(z, xdesired)
L1 = ((z - xdata[0]) * (z - xdata[2])) / ((xdata[1] - xdata[0]) * (xdata[1] - xdata[2]))
L11 = L1.subs(z, xdesired)
L2 = ((z - xdata[0]) * (z - xdata[1])) / ((xdata[2] - xdata[0]) * (xdata[2] - xdata[1]))
L21 = L2.subs(z, xdesired)
fq = L0 * ydata[0] + L1 * ydata[1] + L2 * ydata[2]
fxdesired = fq.subs(z, xdesired)

fnew = fxdesired
ea = np.absolute((fnew - fprev) / fnew * 100)
print((fnew - fprev))
if ea >= 5:
    sd = 0
else:
    sd = floor(2 - np.log10(np.abs(ea) / 0.5))
# displaying the outputs:
print('---------------------------------------------------------------------')
print('\nQUADRATIC INTERPOLATION:')
print('\nx data chosen: x1 = {}, x2 = {}, x3 = {}'.format(xdata[0], xdata[1], xdata[2]))
print('\ny data chosen: y1 = {}, y2 = {}, y3 = {}'.format(ydata[0], ydata[1], ydata[2]))

# displaying the values for the Lagrangian polynomial.
print('\nLagrangian weight function values, L0 = {}, L1 = {}, L2 = {}'.format(L01, L11, L21))

# Using concactenation to properly printlay the final function inside the
# command window.
print('\n')
str1 = 'Final function, f(x) = ', str(L01), '(', str(ydata[0]), ')'
if L11 > 0:
    str2 = ' + ', str(L11), '(', str(ydata[1]), ')'
else:
    str2 = ' ', str(L11), '(', str(ydata[1]), ')'

if L21 > 0:
    str3 = ' + ', str(L21), '(', str(ydata[2]), ')'
else:
    str3 = ' ', str(L21), '(', str(ydata[2]), ')'
#  Putting all the strings together.
finalstr = str1 + str2 + str3
print(finalstr)
# displaying the final value of f(x) at xdesired.
print('\nThe value of f(x) at xdesired is, f({} = {}'.format(xdesired, fxdesired))
# Calculating the absolute relative approximate error and significant
# digits.
print('\nAbsolute Relative Approximate Error, abrae = {}'.format(ea))
print('\nNumber of significant digits at least correct, sig_dig = {}'.format(sd))

# Setting up the parameters for plotting.
z1 = np.arange(np.min(xdata), np.max(xdata), 0.1)
fz = [fl.subs(z, zz) for zz in z1]

# displaying the graphs for Quadratic interpolation.
fig, ax = plt.subplots(2, figsize=(20, 10))  # Create a figure containing a single axes.
print('z1=', fz)
ax[0].plot(z1, fz)
ax[0].set_title('Quadratic Interpolation (Data Points Used)')
ax[0].set_ylabel('f(x)')
ax[0].set_xlabel('x data')
ax[0].plot(xdata, ydata, 'ro')
ax[0].plot(xdesired, fxdesired, 'kx')
#  Creating the second plot for Quadratic Interpolation.
ax[1].plot(z1, fz)
ax[1].set_title('Quadratic Interpolation (Full Data Set)')
ax[1].set_ylabel('f(x)')
ax[1].set_xlabel('x data')
ax[1].plot(x, y, 'ro')
ax[1].plot(xdesired, fxdesired, 'kx')
ax[1].set_xlim([np.min(x), np.max(x)])
ax[1].set_ylim([np.min(y), np.max(y)])
fig.show()
fprev = fnew

######### CUBIC INTERPOLATION #########

# Pick four data points
datapoints = 4
p = 0
xdata = np.zeros(datapoints)
ydata = np.zeros(datapoints)
for i in range(n - 1):
    if p >= datapoints:
        break
    if d[i] <= datapoints:
        xdata[p] = x[i]
        ydata[p] = y[i]
        p += 1

# Calculating coefficients of Newton's Divided difference polynomial
z = symbols('z')
L0 = ((z - xdata[1]) * (z - xdata[2]) * (z - xdata[3])) / (
        (xdata[0] - xdata[1]) * (xdata[0] - xdata[2]) * (xdata[0] - xdata[3]))
L01 = L0.subs(z, xdesired)
L1 = ((z - xdata[0]) * (z - xdata[2]) * (z - xdata[3])) / (
        (xdata[1] - xdata[0]) * (xdata[1] - xdata[2]) * (xdata[1] - xdata[3]))
L11 = L1.subs(z, xdesired)
L2 = ((z - xdata[0]) * (z - xdata[1]) * (z - xdata[3])) / (
        (xdata[2] - xdata[0]) * (xdata[2] - xdata[1]) * (xdata[2] - xdata[3]))
L21 = L2.subs(z, xdesired)
L3 = ((z - xdata[0]) * (z - xdata[1]) * (z - xdata[2])) / (
        (xdata[3] - xdata[0]) * (xdata[3] - xdata[1]) * (xdata[3] - xdata[2]))
L31 = L3.subs(z, xdesired)

# Calculating the final value for the Lagrangian polynomial at "xdesired".
fc = L0 * ydata[0] + L1 * ydata[1] + L2 * ydata[2] + L3 * ydata[3]
fxdesired = fc.subs(z, xdesired)

fnew = fxdesired
ea = abs((fnew - fprev) / fnew * 100)

# Calculating the correct number of significant digits to be taken based on
# the absolute relative approximate error.
if ea >= 5:
    sd1 = 0
else:
    sd1 = floor(2 - log10(abs(ea) / 0.5))
# displaying the outputs:
print('---------------------------------------------------------------------')
print('\nCUBIC INTERPOLATION:')
print('\nx data chosen: x1 = {}, x2 = {}, x3 = {}, x4 = {}'.format(xdata[0], xdata[1], xdata[2], xdata[3]))
print('\ny data chosen: y1 = {}, y2 = {}, y3 = {}, y4 = {}'.format(ydata[0], ydata[1], ydata[2], ydata[3]))

# displaying the values for the Lagrangian polynomial.
print('\nLagrangian weight function values, L0 = {}, L1 = {}, L2 = {}, L3 = {}'.format(L01, L11, L21, L31))

# Using concactenation to properly printlay the final function inside the
# command window.
str1 = 'Final function, f(x) = ', str(L01), '(', str(ydata[0]), ')'
if L11 > 0:
    str2 = ' + ', str(L11), '(', str(ydata[1]), ')'
else:
    str2 = ' ', str(L11), '(', str(ydata[1]), ')'

if L21 > 0:
    str3 = ' + ', str(L21), '(', str(ydata[2]), ')'
else:
    str3 = ' ', str(L21), '(', str(ydata[2]), ')'

if L31 > 0:
    str4 = ' + ', str(L31), '(', str(ydata[3]), ')'
else:
    str4 = ' ', str(L31), '(', str(ydata[3]), ')'
# Putting all the strings together.
finalstr = str1 + str2 + str3 + str4
print(finalstr)

# displaying the value of f(x) at xdesired.
print('\nThe value of f(x) at xdesired is, f({}) = {}'.format(xdesired, fxdesired))
# displaying the number of significant digits that can be taken and
# absolute relative approximate error.
print('\nAbsolute relative approximate error, abrae = {}%'.format(ea))
print('\nNumber of significant digits that can be taken, sig_dig = {}'.format(sd1))

# Setting up the parameters for plotting.
z1 = np.arange(np.min(xdata), np.max(xdata), 0.1)
fz = [fl.subs(z, zz) for zz in z1]

# displaying the graphs for Cubic interpolation.
fig, ax = plt.subplots(2, figsize=(20, 10))  # Create a figure containing a single axes.
print('z1=', fz)
ax[0].plot(z1, fz)
ax[0].set_title('Cubic Interpolation (Data Points Used)')
ax[0].set_ylabel('f(x)')
ax[0].set_xlabel('x data')
ax[0].plot(xdata, ydata, 'ro')
ax[0].plot(xdesired, fxdesired, 'kx')
#  Creating the second plot for Cubic Interpolation.
ax[1].plot(z1, fz)
ax[1].set_title('Cubic Interpolation (Full Data Set)')
ax[1].set_ylabel('f(x)')
ax[1].set_xlabel('x data')
ax[1].plot(x, y, 'ro')
ax[1].plot(xdesired, fxdesired, 'kx')
ax[1].set_xlim([np.min(x), np.max(x)])
ax[1].set_ylim([np.min(y), np.max(y)])
fig.show()
