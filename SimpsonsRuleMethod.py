# Mfile name
#       mtl_int_sim_simpsonmeth.m

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
#       To illustrate simpson's 1/3rd rule as applied to a function
#       of the user's choosing.

# Keyword
#       Simpson's Rule
#       Numerical Integration

# Inputs
#       This is the only place in the program where the user makes the changes
#       based on their wishes

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# f(x), the function to integrate

f = lambda x: 2000 * np.log(1400 / 21. / x) - 9.8 * x

# a, the lower limit of integration

a = 8

# b, the upper limit of integration

b = 12

# n, the number of segments. Note that this number must be even.

n = 12

# **********************************************************************

# This printlays title information
print('\n\nSimulation of the Simpson''s 1/3rd Rule')
print('University of South Florida')
print('United States of America')
print('kaw@eng.usf.edu\n')

print('\n*******************************Introduction*********************************')

# printlays introduction text
print('Trapezoidal rule was based on approximating the integrand by a first order')
print('polynomial, and then integrating the polynomial in the interval of integration.')
print('Simpson''s 1/3 rule is an extension of Trapezoidal rule where the integrand is')
print('now approximated by a second order polynomial.')

# printlays what inputs will be used
print('\n\n********************************Input Data**********************************\n')
print('     f(x), integrand function')
print('     a = {}, lower limit of integration '.format(a))
print('     b = {}, upper limit of integration '.format(b))
print('     n = {}, number of subdivisions (must be even)'.format(n))

print('\nFor this simulation, the following parameter is constant.\n')
# calculate the spacing parameter.
h = (b - a) / n
print('     h = ( b - a ) / n ')
print('       = ( {} - {} ) / {} '.format(b, a, n))
print('       = {}'.format(h))

# This begins the simulation of the method
print('\n*******************************Simulation*********************************\n')
sum = 0
print('The approximation is expressed as')
print(' ')
print('     approx = h/3 * ( f(a) + 4*Sum(i=1,n-1,odd) f(a+i*h) +')
print('                             2*Sum(i=2,n-2,even) f(a+i*h) + f(b) )')

# Sum the odd function values and multiply by 4
print(' ')
print('1) Begin summing odd function values at points between a and b not')
print('   including a and b. Multiply this by 4.')
print(' ')
print('       4*Sum (i=1,n-1,odd) f(a+i*h)')
print(' ')
# sum of functions evaluated at odd spacings
for i in range(0, n - 4, 2):
    print('          f({})'.format(a + i * h))

print('       +  f({})'.format(a + (n - 1) * h))
print('       ------------')
for i in range(0, n - 4, 2):
    sum = sum + f(a + i * h)
    print('          {}'.format(f(a + i * h)))

sum = sum + f(a + (n - 1) * h)
print('       +  {}'.format(f(a + (n - 1) * h)))
print('       ------------')
print('       4* {} = {}\n'.format(sum, 4 * sum))
sum = 4 * sum
# Sum the even function values, and multiply by 2
sum2 = 0
print(' ')
print('2) Continue by summing even function values at points between a and')
print('   b not including a and b. Multiply this sum by 2.')
print(' ')
print('       2*Sum (i=1,n-1,even) f(a+i*h)')
print(' ')

for i in range(1, n - 5, 2):
    print('          f({})'.format(a + i * h))
print('       +  f({})'.format(a + (n - 2) * h))
print('       ------------')
# sum of functions evaluated at even spacings
for i in range(1, n - 5, 2):
    sum2 = sum2 + f(a + i * h)
    print('          {}'.format(f(a + i * h)))

sum2 = sum2 + f(a + (n - 2) * h)
print('       +  {}'.format(f(a + (n - 2) * h)))
print('       ------------')
print('       2* {} = {}\n'.format(sum2, 2 * sum2))
sum2 = 2 * sum2
# Add the two sumns
print('3) Add f(a) and f(b) to the sums from 1) and 2)')
print(' ')
print('     {} + {} + {} + {} = {}'.format(f(a), sum, sum2, f(b), f(a) + sum + sum2 + f(b)))
sum = sum + sum2 + f(a) + f(b)
# Then multiply by h/3
print(' ')
print('4) Multiply this by h/3 to get the approximation for the integral.')
print(' ')
print('     approx = h/3 * {}'.format(sum))
print('     approx = {} * {}'.format(h / 3, sum))
approx = h / 3 * sum
print('     approx = {}'.format(approx))

# Display the results
print('\n\n**************************Results****************************')

# The following finds what is called the 'Exact' solution
exact = quad(f, a, b)

print('\n   Approximate = {}'.format(approx))
print('   Exact       = {}'.format(exact))
print('\n   True Error = Exact - Approximate')
print('              = {} - {}'.format(exact, approx))
print('              = {}'.format(exact - approx))
print('\n   Absolute Relative True Error Percentage')
print('              = | ( Exact - Approximate ) / Exact | * 100')
print('              = | {} / {} | * 100'.format(exact - approx, exact))
print('              = {}'.format(abs((exact - approx) / exact) * 100))

print('\nThe Simpson 1/3rd rule can be more accurate if we made our')
print('segment size smaller (that is, increasing the number of segments).\n\n')
# The following code sets up the graphical depiction of the method.
x = np.zeros(n + 1)
y = np.zeros(n + 1)

x[0] = a
y[0] = f(a)

for i in range(n):
    x[i + 1] = a + i + 1 * h
    y[i + 1] = f(x[i + 1])
    print(x[i + 1], y[i + 1])

# polyfit is used to fit quadratics to each interval of 3 points.
# Simpson's rule is equivalent to these approximations integrated.
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
for i in range(0, n, 2):
    bg = i
    ed = i + 2
    coeffs = np.polyfit(x[bg:ed], y[bg:ed], 2)
    poly_x1 = np.arange(x[bg], x[ed], (x[ed] - x[bg]) / 1000)

    poly_y1 = np.multiply(coeffs[0], np.power(poly_x1, 2)) + np.multiply(coeffs[1], poly_x1) + coeffs[2]
    poly_x1 = np.hstack([poly_x1[0], poly_x1, poly_x1[-1]])
    poly_y1 = np.hstack([0, poly_y1, 0])

    ax.fill(poly_x1, poly_y1, 'y', edgecolor='black', linewidth=2)

xrange = np.arange(a, b, (b - a) / 1000)
ax.plot(xrange, f(xrange), 'k')
ax.set_title('Integrand function and Graphical Depiction of Simpson''s 1/3rd Rule')
fig.show()
