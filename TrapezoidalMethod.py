# Revised:
#       March 7, 2008

# # Authors
#       Nathan Collier, Autar Kaw
#       Department of Mechanical Engineering
#       University of South Florida
#       Tampa, Fl 33620
#       Contact: kaw@eng.usf.edu | http://numericalmethods.eng.usf.edu/contact.html
# Translate  : Furkan Kıyıkçı
# Purpose
#       To illustrate the trapezoidal method as applied to a function
#       of the user's choosing.

# Keyword
#       Trapezoidal Method
#       Numerical Integration

# Inputs
#       This is the only place in the program where the user makes the changes
#       based on their wishes
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# f(x), the function to integrate
f = lambda x: 2000 * np.log(1400 / 21. / x) - 9.8 * x

# a, the lower limit of integration

a = 8

# b, the upper limit of integration

b = 10

# n, the maximum number of segments

n = 10

# **********************************************************************

# This printlays title information
print('\n\nSimulation of the Trapezoidal Method')
print('University of South Florida')
print('United States of America')
print('kaw@eng.usf.edu\n')
print('\n*******************************Introduction*********************************')
# printlays introduction text
print('Trapezoidal rule is based on the Newton-Cotes formula that if one approximates the')
print('integrand by an nth order polynomial, then the integral of the function is approximated')
print('by the integral of that nth order polynomial. Integrating polynomials is simple and is')
print('based on calculus. Trapezoidal rule is the area under the curve for a first order')
print('polynomial (straight line).')

# printlays what inputs are being used
print('\n\n********************************Input Data**********************************\n')
print('     f(x), integrand function')
print('     a = {}, lower limit of integration '.format(a))
print('     b = {}, upper limit of integration '.format(b))

# Calculate the spacing parameter
print('\nFor this simulation, the following parameter is constant.\n')
h = (b - a) / n
print('     h = ( b - a ) / n ')
print('       = ( {} - {} ) / {} '.format(b, a, n))
print('       = {}'.format(h))

# This begins the simulation of the method
print('\n*******************************Simulation*********************************\n')
sum = 0
print('The approximation is expressed as')
print(' ')
print('     approx = h * ( 0.5*f(a) + Sum (i=1,n-1) f(a+i*h) + 0.5*f(b) )')
print(' ')

# Sum all function values not including evalations at a and b
print('1) Begin summing all function values at points between a and b not')
print('   including a and b.')
print(' ')
print('       Sum (i=1,n-1) f(a+i*h)')
print(' ')

for i in range(n - 3):
    print('          f({})'.format(a + i * h))

print('       +  f({})'.format(a + (n - 1) * h))
print('       ------------')
for i in range(n - 3):
    sum = sum + f(a + i * h)
    print('          {}'.format(f(a + i * h)))
sum = sum + f(a + (n - 1) * h)
print('       +  {}'.format(f(a + (n - 1) * h)))
print('       ------------')
print('          {}\n'.format(sum))

# Now add half the end point evaluations
print('2) Add to this 0.5*(f(a) + f(b))')
print(' ')
print('     {} + 0.5*({} + {}) ='.format(sum, f(a), f(b)))
print('     {} + {} = {}'.format(sum, 0.5 * (f(a) + f(b)), sum + 0.5 * (f(a) + f(b))))
sum = sum + 0.5 * (f(a) + f(b))
print(' ')

# And finally multiply by h
print('3) Multiply this by h to get the approximation for the integral.')
print(' ')
print('     approx = h * {}'.format(sum))
print('     approx = {} * {}'.format(h, sum))
approx = h * sum
print('     approx = {}'.format(approx))

# The following displays results
print('\n\n**************************Results****************************')

# The following finds what is called the 'Exact' solution
exact = quad(f, a, b)
print('\n   Approximate = {}'.format(approx))
# print('   Exact       = {}'.format(exact))
print('\n   True Error = Exact - Approximate')
print('              = {} - {}'.format(exact, approx))
print('              = {}'.format(exact - approx))
print('\n   Absolute Relative True Error Percentage')
print('              = | ( Exact - Approximate ) / Exact | * 100')
print('              = | {} / {} | * 100'.format(exact - approx, exact))
print('              = {}'.format(abs((exact - approx) / exact) * 100))

print('\nThe trapezoidal approximation can be more accurate if we made our')
print('segment size smaller (that is, increasing the number of segments).\n\n')

# The following code is needed to produce the trapezoidal method
# visualization.
x = np.zeros(n + 1)
y = np.zeros(n + 1)

x[0] = a
y[0] = f(a)
print('x=', x)
print('y=', y)
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
for i in range(n):
    x[i + 1] = a + (i + 1) * h
    y[i + 1] = f(x[i + 1])

    ax.fill([x[i], x[i], x[i + 1], x[i + 1]], [0, y[i], y[i + 1], 0], 'y', edgecolor='black', linewidth=2)

xrange = np.arange(a, b, (b - a) / 1000)

ax.plot(xrange, f(xrange), 'k')
ax.set_title('Integrand function and Graphical Depiction of Trapezoidal Method')
fig.show()
