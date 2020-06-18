# Mfile name
#       mtl_int_sim_eulermethod.m

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
#       To illustrate Euler's method applied
#       to a function of the user's choosing.

# Keyword
#       Euler's Method
#       Convergence
#       Ordinary Differential Equations
#       Initial Value Problem

# Inputs
#       This is the only place in the program where the user makes the changes
#       based on their wishes

# dy/dx in form of f(x,y). In general it can be a function of both 
# variables x and y. If your function is only a function of x then
# you will need to add a 0*y to your function.
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

f = lambda x, y: np.exp(-x) + 0 * y
g = lambda y, x: np.exp(-x) + 0 * y

# x0, x location of known initial condition

x0 = 3

# y0, corresponding value of y at x0

y0 = 2

# xf, x location at where you wish to see the solution to the ODE

xf = 12

# n, number of steps to take

n = 10

# **********************************************************************

# Displays title information
print('\n\nEuler Method of Solving Ordinary Differential Equations')
print('University of South Florida')
print('United States of America')
print('kaw@eng.usf.edu\n')
print('NOTE: This worksheet demonstrates the use of Matlab to illustrate ')
print('Euler''s method, a numerical technique in solving ordinary differential')
print('equations.')

# Displays introduction text
print('\n***************************Introduction****************************')
print('Euler''s method approximates the solution to an ordinary differential')
print('equation by using the equation expressed in the form dy/dx = f(x,y) to')
print('approximate the slope. This slope is used to project the solution to')
print('the ODE a fixed distance away.')

# Displays inputs being used
print('\n\n****************************Input Data*****************************')
print('     f = dy/dx ')
print('     x0 = initial x ')
print('     y0 = initial y ')
print('     xf = final x ')
print('     n = number of steps to take')

print('\n-----------------------------------------------------------------\n')
print('[     f(x,y) = dy/dx = {} ]'.format('exp(-x)+0*y'))
print('     x0 = {}'.format(x0))
print('     y0 = {}'.format(y0))
print('     xf = {}'.format(xf))
print('     n = {}'.format(n))
print('\n-----------------------------------------------------------------')
print('For this simulation, the following parameter is constant.\n')
h = (xf - x0) / n
print('     h = ( xf - x0 ) / n ')
print('       = ( {} - {} ) / {} '.format(xf, x0, n))
print('       = {}'.format(h))
xa = np.zeros(n+1)
ya = np.zeros(n+1)
xa[0] = x0
ya[0] = y0

# Here begins the method
print('\n\n**************************Simulation*****************************')

for i in range(n):
    print('\nStep {}'.format(i+1))
    print('-----------------------------------------------------------------')

    # Adding Step Size
    xa[i + 1] = xa[i] + h

    # Calculating f(x,y) at xa(i) and ya(i)
    fcn = f(xa[i], ya[i])

    # Using Euler's formula
    ya[i + 1] = ya[i] + fcn * h

    print('1) Evaluate the function f at the previous, values of x and y.')
    print('     f( x{} , y{} ) = f( {} , {} ) = {}'.format(i - 1, i - 1, xa[i], ya[i], fcn))
    print('2) Apply the Euler method to estimate y{}'.format(i))
    print('     y{} = y{} + f( x{}, y{} ) * h '.format(i, i - 1, i - 1, i - 1))
    print('        = {} + {} * {} '.format(ya[i], fcn, h))
    print('        = {}'.format(ya[i + 1]))
    print('   at x{} = {}'.format(i, xa[i + 1]))

# The following are the results 
print('\n\n**************************Results****************************')

# The following finds what is called the 'Exact' solution
xspan = np.linspace(x0, xf, 16)
Us = odeint(g, y0, xspan)
y = Us[:, 0]
yf= y[-1]
# Plotting the Exact and Approximate solution of the ODE.
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Exact and Approximate Solution of the ODE by Euler''s Method')
ax.plot(xspan, y, 'b--', linewidth=2, label='Exact')
ax.plot(xa, ya, '-', linewidth=2, color=[0, 1, 0], label='Approximation')
ax.legend(loc=1)  # Add a legend.
fig.show()
print('The figure window that now appears shows the approximate solution as ')
print('piecewise continuous straight lines. The blue line represents the exact')
print('solution. In this case ''exact'' refers to the solution obtained by the')
print('Matlab function ode45.\n')

print('While Euler''s method is valid for approximating the solutions of')
print('ordinary differential equations, the use of the slope at one point')
print('to project the value at the next point is not very accurate. Note the')
print('approximate value obtained as well as the true value and relative true')
print('error at our desired point x = xf.')

print('\n   Approximate = {}'.format(ya[-1]))
print('   Exact       = {}'.format(yf))
print('\n   True Error = Exact - Approximate')
print('              = {} - {}'.format(yf, ya[-1]))
print('              = {}'.format(yf - ya[-1]))
print('\n   Absolute Relative True Error Percentage')
print('              = | ( Exact - Approximate ) / Exact | * 100')
print('              = | {} / {} | * 100'.format(yf - ya[-1], yf))
print('              = {}'.format(abs((yf - ya[-1]) / yf) * 100))

print('\nThe Euler approximation can be more accurate if we made our')
print('step size smaller (that is, increasing the number of steps).\n\n')
