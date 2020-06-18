# Mfile name
#       mtl_int_sim_RK2ndmethod.m

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
#       To illustrate the 2nd order Runge-Kutta method applied
#       to a function of the user's choosing.

# Keyword
#       Runge-Kutta
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

f = lambda x, y: y * x ** 2 - 1.2 * y
g = lambda y, x: y * x ** 2 - 1.2 * y

# x0, x location of known initial condition

x0 = 0

# y0, corresponding value of y at x0

y0 = 1

# xf, x location at where you wish to see the solution to the ODE

xf = 2

# a2, parameter which must be between 0 and 1. Certain names are associated
# with different parameters.
# 
# a2 = 0.5 Heun's Method
#    = 2/3 Ralston's Method
#    = 1.0 Midpoint Method

a2 = 0.5

# n, number of steps to take

n = 5

# **********************************************************************

# Displays title information
print('\n\nThe 2nd Order Runge-Kutta Method of Solving Ordinary Differential Equations')
print('University of South Florida')
print('United States of America')
print('kaw@eng.usf.edu\n')
print('NOTE: This worksheet demonstrates the use of Matlab to illustrate ')
print('the Runge-Kutta method, a numerical technique in solving ordinary ')
print('differential equations.')

print('\n***************************Introduction*******************************')

# Displays introduction text 
print('The 2nd Order Runge-Kutta method approximates the solution to an ')
print('ordinary differential equation by using the equation expressed in')
print('the form dy/dx = f(x,y) to approximate the slope. This slope is used')
print('to project the solution to the ODE a fixed distance away.')

# displays what inputs are used
print('\n\n***************************Input Data*******************************')
print('Below are the input parameters to begin the simulation which can be')
print('changed in the m-file input section')
print('\n     f = dy/dx ')
print('     x0 = initial x ')
print('     y0 = initial y ')
print('     xf = final x ')
print('     a2 = constant value between 0 and 1.')
print('        = 0.5, Heun Method')
print('        = 1.0, Midpoint Method')
print('        = 0.66667, Ralston''s Method')
print('      n = number of steps to take')
print('\n-----------------------------------------------------------------\n')
print('     f(x,y) = dy/dx = '.format('y*x^2-1.2*y'))
print('     x0 = {}'.format(x0))
print('     y0 = {}'.format(y0))
print('     xf = {}'.format(xf))
print('     a2 = {}'.format(a2))
print('     n = {}'.format(n))
print('\n-----------------------------------------------------------------')
print('For this simulation, the following parameter is constant.\n')

# Calculates constants used in the method
h = (xf - x0) / n
print('     h = ( xf - x0 ) / n ')
print('       = ( {} - {} ) / {} '.format(xf, x0, n))
print('       = {}'.format(h))
a1 = 1 - a2
print('\n    a1 = 1 - a2')
print('       = 1 - {}'.format(a2))
print('       = {}'.format(a1))
p1 = 1 / 2 / a2
print('\n    p1 = 1 / ( 2 * a2 )')
print('       = 1 / ( 2 * {} )'.format(a2))
print('       = {}'.format(p1))
q11 = p1
print('\n   q11 = p1')
print('       = {}'.format(q11))

xa = np.zeros(n+1)
ya = np.zeros(n+1)
xa[0] = x0
ya[0] = y0

print('\n\n***************************Simulation******************************')

for i in range(n ):
    print('\nStep {}'.format(i))
    print('-----------------------------------------------------------------')

    # Adding Step Size
    xa[i + 1] = xa[i] + h

    # Calculating k1 and k2
    k1 = f(xa[i], ya[i])
    k2 = f(xa[i] + p1 * h, ya[i] + q11 * k1 * h)

    # Using 2nd Order Runge-Kutta formula
    ya[i + 1] = ya[i] + (a1 * k1 + a2 * k2) * h

    print('1) Find k1 and k2 using the previous step information.')
    print('     k1 = f( x{} , y{} )'.format(i - 1, i - 1))
    print('        = f( {} , {} ) )'.format(xa[i], ya[i]))
    print('        = {}\n'.format(k1))
    print('     k2 = f( x{} + p1 * h , y{} + q11 * k1 * h )'.format(i - 1, i - 1))
    print('        = f( {} + {} * {} , {} + {} * {} * {})'.format(xa[i], p1, h, ya[i], q11, k1, h))
    print('        = f( {} , {} )'.format(xa[i] + p1 * h, ya[i] + q11 * k1 * h))
    print('        = {}\n'.format(k2))

    print('2) Apply the Runge-Kutta 2nd Order method to estimate y{}'.format(i))
    print('     y{} = y{} + ( a1 * k1 + a2 * k2 ) * h'.format(i, i - 1))
    print('        = {} + {} * {}'.format(ya[i], a1 * k1 + a2 * k2, h))
    print('        = {}\n'.format(ya[i + 1]))
    print('   at x{} = {}'.format(i, xa[i + 1]))

print('\n\n********************************Results**********************************')

# The following finds what is called the 'Exact' solution
xspan = np.arange(x0, xf, 0.1)
Us = odeint(g, y0, xspan)
y = Us[:, 0]
yf = y[-1]

# Plotting the Exact and Approximate solution of the ODE.

fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Exact and Approximate Solution of the ODE by the 2nd Order Runge-Kutta Method')
ax.plot(xspan, y, 'b--', linewidth=2, label='Exact')
ax.plot(xa, ya, '-', linewidth=2, color=[0, 1, 0], label='Approximation')
ax.legend(loc=1)  # Add a legend.
fig.show()
print('The figure window that now appears shows the approximate solution as ')
print('piecewise continuous straight lines. The blue line represents the exact')
print('solution. In this case ''exact'' refers to the solution obtained by the')
print('Matlab function ode45.\n')

print('Note the approximate value obtained as well as the true value and ')
print('relative true error at our desired point x = xf.')

print('\n   Approximate = {}'.format(ya[-1]))
print('   Exact       = {}'.format(yf))
print('\n   True Error = Exact - Approximate')
print('              = {} - {}'.format(yf, ya[-1]))
print('              = {}'.format(yf - ya[-1]))
print('\n   Absolute Relative True Error Percentage')
print('              = | ( Exact - Approximate ) / Exact | * 100')
print('              = | {} / {} | * 100'.format(yf - ya[-1], yf))
print('              = {}'.format(abs((yf - ya[-1]) / yf) * 100))

print('\nThe approximation can be more accurate if we made our step')
print('size smaller (that is, increasing the number of steps).\n\n')
