# Mfile name
#       mtl_int_sim_shootingmethod.m

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
#       To illustrate the use of Runge-Kutta method in the Shooting method as applied
#       to a function of the user's choosing.

# Keyword
#       Runge-Kutta
#       Shooting Method
#       Ordinary Differential Equations
#       2nd Order Boundary Value Problem

# Inputs
#       This is the only place in the program where the user makes the changes
#       based on their wishes

# Define the differential equation to be solved for of the form
# 
#      A1*d/dx(dy/dx) + A2*dy/dx + A3*y + A4 = 0
#
# The coefficients a1, a2, and a3 can be functions of x and y while
# a4 can be a function of x only. Define these functions here. 
import numpy as np
import matplotlib.pyplot as plt

A1 = lambda x, y: 2
A2 = lambda x, y: 4
A3 = lambda x, y: 6
A4 = lambda x: 8
# Create functions 1 and 2 for use in the method

f1 = lambda x, y, z: z
f2 = lambda x, y, z: 1 / A1(x, y) * (A2(x, y) * z + A3(x, y) * y + A4(x))

# x0, x location of known initial condition

x0 = 0

# y0, corresponding value of y at x0

y0 = 10

# xf, x location of known boundary condition

xf = 1

# yf, corresponding value of y at x0

yf = 0

# dydx1, initial guess of the derivative dy/dx at x = x0

dydx1 = -15

# dydx2, initial guess of the derivative dy/dx at x = x0

dydx2 = -10

# n, number of steps to take

n = 10
# **********************************************************************

# Displays title information
print('\n\nShooting Method of Solving Ordinary Differential Equations')
print('University of South Florida')
print('United States of America')
print('kaw@eng.usf.edu')
print('NOTE: This worksheet demonstrates the use of Matlab to illustrate the')
print('shooting method (by means of the 4th Order Runge-Kutta method) to solve')
print('higher order ODE''s with displacement boundary conditions.')

print('\n*******************************Introduction*********************************')
# Displays introduction text
print(
    'The shooting method of solving ODEs is used when there are multiple initial \nconditions to be satisfied at different independent variable locations. For \nexample, as in many real life problems such as a simply supported beam, \nthe boundary conditions are y(0) = 0 and y(L) = 0 where y represents the \ndisplacement and L is the beam length. The methods for solving ODE''s \nbegin at one point and using approximations of the derivative, find the \nsolution from one end to another. These methods will not take into account \nthe boundary condition on the other end. The shooting method uses a method \nof solving ODEs (4th order Runge-Kutta in this example) in a way that \nsatisfies both boundary conditions')

# Displays inputs used
print('\n\n****************************Input Data******************************')
print('Below are the input parameters to begin the simulation. Although this method')
print('can be used to solve ODE''s of any order, the ODE being approximated here is')
print('of second order.')
print('\n     A1*d/dx(dy/dx) + A2*(dy/dx) + A3*y + A4 = 0\n')
print('This second order equation can be written in terms of 2 first order equations.')
print('The user must be able to break up his ODE into these two functions.')
print('\n     f1 = dy/dx = z    ')
print('     f2 = dz/dx = 1/A1 * ( A2*z + A3*y + A4 )')
print('     x0 = initial x')
print('     y0 = initial y')
print('     xf = final x')
print('     yf = final y')
print('     dydx1 = 1st guess of derivative at x = x0')
print('     dydx2 = 2nd guess of derivative at x = x0')
print('     n = number of steps to take')

print('\n-----------------------------------------------------------------')
print('\n     x0 = {}'.format(x0))
print('     y0 = {}'.format(y0))
print('     xf = {}'.format(xf))
print('     yf = {}'.format(yf))
print('     dydx1 = {}'.format(dydx1))
print('     dydx2 = {}'.format(dydx2))
print('     n = {}'.format(n))
print('\n-----------------------------------------------------------------')
print('For this simulation, the following parameters are constant.')

# compute the spacing
h = (xf - x0) / n
print('\n     h = ( xf - x0 ) / n')
print('       = ( {} - {} ) / {}'.format(xf, x0, n))
print('       = {}'.format(h))

# determine the x value for each step
x = np.zeros(n+1)

for i in range(n+1):
    x[i] = x0 + i * h

# The simulation begins here.
print('\n\n********************************Simulation**********************************')
print('\nIteration 1')
print('-----------------------------------------------------------------')
print('The first step is to use the 4th Order Runge-Kutta method to solve the')
print('problem using the first guess for the derivative (dydx1) and only the left')
print('boundary conditions (x0,y0). If the initial guess for the derivative was right,')
print('then the approximation should end up exactly on yf. The following shows the')
print('result of the 4th Order Runge-Kutta method and compares it to the second')
print('boundary condition.')

# Set initial conditions and pick first derivative as initial condition.
y1 = np.zeros(n+1)
z1 = np.zeros(n+1)
y1[0] = y0
z1[0] = dydx1

# Application of 4th order Runge-Kutta to solve higher order ODE's
for i in range(n):
    k1y = f1(x[i], y1[i], z1[i])
    k1z = f2(x[i], y1[i], z1[i])
    k2y = f1(x[i] + 0.5 * h, y1[i] + 0.5 * k1y * h, z1[i] + 0.5 * k1z * h)
    k2z = f2(x[i] + 0.5 * h, y1[i] + 0.5 * k1y * h, z1[i] + 0.5 * k1z * h)
    k3y = f1(x[i] + 0.5 * h, y1[i] + 0.5 * k2y * h, z1[i] + 0.5 * k2z * h)
    k3z = f2(x[i] + 0.5 * h, y1[i] + 0.5 * k2y * h, z1[i] + 0.5 * k2z * h)
    k4y = f1(x[i] + h, y1[i] + k3y * h, z1[i] + k3z * h)
    k4z = f2(x[i] + h, y1[i] + k3y * h, z1[i] + k3z * h)
    y1[i + 1] = y1[i] + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    z1[i + 1] = z1[i] + h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)

print('\n     y1(xf) = {}'.format(y1[-1]))
print('     yf = {}\n'.format(yf))

print('\nIteration 2')
print('-----------------------------------------------------------------')
print('Most likely the first guess of the derivative was not correct. So we repeat')
print('this process using the second guess for the derivative (dydx2). Again, we')
print('can compare the results with the boundary condition.')
y2 = np.zeros(n+1)
z2 = np.zeros(n+1)
y2[0] = y0
z2[0] = dydx2
for i in range(n):
    k1y = f1(x[i], y2[i], z2[i])
    k1z = f2(x[i], y2[i], z2[i])
    k2y = f1(x[i] + 0.5 * h, y2[i] + 0.5 * k1y * h, z2[i] + 0.5 * k1z * h)
    k2z = f2(x[i] + 0.5 * h, y2[i] + 0.5 * k1y * h, z2[i] + 0.5 * k1z * h)
    k3y = f1(x[i] + 0.5 * h, y2[i] + 0.5 * k2y * h, z2[i] + 0.5 * k2z * h)
    k3z = f2(x[i] + 0.5 * h, y2[i] + 0.5 * k2y * h, z2[i] + 0.5 * k2z * h)
    k4y = f1(x[i] + h, y2[i] + k3y * h, z2[i] + k3z * h)
    k4z = f2(x[i] + h, y2[i] + k3y * h, z2[i] + k3z * h)
    y2[i + 1] = y2[i] + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    z2[i + 1] = z2[i] + h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)

print('\n     y2(xf) = {}\n     yf = {}\n'.format(y2[-1], yf))

print('\nIteration 3')
print('-----------------------------------------------------------------')

y3 = np.zeros(n+1)
z3 = np.zeros(n+1)
# Linearly interpolate to obtain another estimate for the derivative at x = x0    
dydx3 = (dydx1 - dydx2) / (y1[-1] - y2[-1]) * (yf - y2[-1]) + dydx2
y3[0] = y0
z3[0] = dydx3

print('Now we can use these two results to find a new approximation for the')
print('derivative. We can linearly interpolate to find and approximation for dydx3.')
print('This will certainly get the next guess closer, although how close will depend')
print('on the character on the ODE.')
print('\n     dydx3 = ( dydx1 - dydx2 )/( y1(xf) - y2(xf) )*( yf - y2(xf) ) + dydx2')
print('     dydx3 = ( {} - {} )/( {} - {} )*( {} - {} ) + {}'.format(dydx1, dydx2, y1[-1], y2[-1], yf, y2[-1],
                                                                     dydx2))
print('           = {}\n'.format(dydx3))

for i in range(n):
    k1y = f1(x[i], y3[i], z3[i])
    k1z = f2(x[i], y3[i], z3[i])
    k2y = f1(x[i] + 0.5 * h, y3[i] + 0.5 * k1y * h, z3[i] + 0.5 * k1z * h)
    k2z = f2(x[i] + 0.5 * h, y3[i] + 0.5 * k1y * h, z3[i] + 0.5 * k1z * h)
    k3y = f1(x[i] + 0.5 * h, y3[i] + 0.5 * k2y * h, z3[i] + 0.5 * k2z * h)
    k3z = f2(x[i] + 0.5 * h, y3[i] + 0.5 * k2y * h, z3[i] + 0.5 * k2z * h)
    k4y = f1(x[i] + h, y3[i] + k3y * h, z3[i] + k3z * h)
    k4z = f2(x[i] + h, y3[i] + k3y * h, z3[i] + k3z * h)
    y3[i + 1] = y3[i] + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    z3[i + 1] = z3[i] + h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)

print('Now we can use this new approximation for the derivative (dydx3) to solve')
print('another 4th Order Runge-Kutta problem. Comparing the result with the')
print('boundary condition, we find')

print('\n     y3(xf) = {}'.format(y3[-1]))
print('     yf = {}'.format(yf))
print('     Error = {}\n\n'.format(y3[-1] - yf))

# Plot the three approximations showing the improvement


fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot([x0, xf], [y0, yf], 'ko', markersize=10, linewidth=2, label='Boundary Conditions')
ax.plot(x, y1, 'r-', linewidth=2, label='1st Iteration')
ax.plot(x, y2, 'g-', linewidth=2, label='2nd Iteration')
ax.plot(x, y3, 'b-', linewidth=2, label='3rd Iteration')
ax.legend(loc=1)  # Add a legend.
fig.show()
