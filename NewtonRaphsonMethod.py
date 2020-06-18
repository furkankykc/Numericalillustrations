# Topic      : Newton-Raphson Method - Roots of Equations
# Simulation : Simulation of the Method
# Language   : Python v3.8.3
# Authors    : Nathan Collier, Autar Kaw
# Translate  : Furkan Kıyıkçı
# Date       : 21 August 2002
# Abstract   : This simulation illustrates the Newton-Raphson method of  
#              finding roots of an equation f(x)=0.
#
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

x = symbols('x')
# INPUTS: Enter the following
# Function in f(x)=0
f = x ** 3 - 0.165 * x ** 2 + 3.993 * 10 ** (-4)

# Initial guess

x0 = 0.05
# Lower bound of range of 'x' to be seen
lxrange = -0.02
# Upper bound of range of 'x' to be seen
uxrange = 0.12
#
# SOLUTION
g = diff(f)
# The following finds the upper and lower 'y' limits for the plot based on the given
#    'x' range in the input section.
maxi = f.subs(x, uxrange)
mini = f.subs(x, lxrange)
for i in np.arange(lxrange, uxrange, (uxrange - lxrange) / 10):
    if f.subs(x, i) > maxi:
        maxi = f.subs(x, i)
    if f.subs(x, i) < mini:
        mini = f.subs(x, i)

tot = maxi - mini
mini = mini - 0.1 * tot
maxi = maxi + 0.1 * tot

xrange = np.arange(lxrange, uxrange, 0.0001)
y = [f.subs(x, xx) for xx in xrange]
# This graphs the function and the line representing the initial guess

fig, ax = plt.subplots(figsize=(15, 10))  # Create a figure containing a single axes.

ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.

ax.plot(xrange, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([x0, x0], [maxi, mini], 'g')
ax.plot([lxrange, uxrange], [0, 0], 'k')
fig.show()

# --------------------------------------------------------------------------------
# Iteration 1
fig = plt.figure(figsize=(15, 10))  # Create a figure containing a single axes.
ax = plt.subplot(212)

ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.
x1 = x0 - f.subs(x, x0) / g.subs(x, x0)
ea = abs((x1 - x0) / x1) * 100
m = -f.subs(x, x0) / (x1 - x0)
b = f.subs(x, x0) * (1 + x0 / (x1 - x0))
lefty = (maxi - b) / m
righty = (mini - b) / m
ax.plot(xrange, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([x0, x0], [maxi, mini], 'g')
ax.plot([x1, x1], [maxi, mini], 'g')
ax.plot([lefty, righty], [maxi, mini], 'r')
ax.plot([lxrange, uxrange], [0, 0], 'k')
ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.

# This portion adds the text and math to the top part of the figure window
upTextList = [
    'Iteration 1',
    'x1 = x0 - (f(x0)/g(x0)) = {}'.format(x1),
    'Absolute relative approximate error',
    'ea = abs((x1 - x0) / x1)*100 = {}%'.format(ea),
]
[plt.figtext(0.1, .9 - i * .03, text) for i, text in enumerate(upTextList)]
fig.show()

# --------------------------------------------------------------------------------
# Iteration 2

x2 = x1 - f.subs(x, x1) / g.subs(x, x1)
ea = abs((x2 - x1) / x2) * 100
m = -f.subs(x, x1) / (x2 - x1)
b = f.subs(x, x1) * (1 + x1 / (x2 - x1))
lefty = (maxi - b) / m
righty = (mini - b) / m
# This graphs the function and two lines representing the two guesses
fig = plt.figure(figsize=(15, 10))  # Create a figure containing a single axes.
ax = plt.subplot(211)
ax.plot(xrange, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([x1, x1], [maxi, mini], 'g')
ax.plot([x2, x2], [maxi, mini], 'g')
ax.plot([lefty, righty], [maxi, mini], 'r')
ax.plot([lxrange, uxrange], [0, 0], 'k')
ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.

# Calculate relative approximate error
ea = abs((x2 - x1) / x2) * 100

# This portion adds the text and math to the bottom part of the figure window
downTextList = [
    'Iteration 2',
    'x2 = x1 - (f(x1)/g(x1)) = {}'.format(x1),
    'Absolute relative approximate error',
    'ea = abs((x2 - x1) / x2)*100 = {}%'.format(ea),
]
[plt.figtext(0.1, .3 - i * .03, text) for i, text in enumerate(downTextList)]
fig.show()

# --------------------------------------------------------------------------------
# Iteration 3

x3 = x2 - f.subs(x, x2) / g.subs(x, x2)
ea = abs((x3 - x2) / x3) * 100
m = -f.subs(x, x2) / (x3 - x2)
b = f.subs(x, x2) * (1 + x2 / (x3 - x2))
lefty = (maxi - b) / m
righty = (mini - b) / m

fig = plt.figure(figsize=(15, 10))  # Create a figure containing a single axes.
ax = plt.subplot(211)
ax.plot(xrange, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([x2, x2], [maxi, mini], 'g')
ax.plot([x3, x3], [maxi, mini], 'g')
ax.plot([lefty, righty], [maxi, mini], 'r')
ax.plot([lxrange, uxrange], [0, 0], 'k')
ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.

# Calculate relative approximate error
ea = abs((x2 - x1) / x2) * 100
# This portion adds the text and math to the bottom part of the figure window
downTextList = [
    'Iteration 3',
    'x3 = x2 - (f(x2)/g(x2)) = {}'.format(x1),
    'Absolute relative approximate error',
    'ea = abs((x3 - x2) / x2)*100 = {}%'.format(ea),
]
[plt.figtext(0.1, .3 - i * .03, text) for i, text in enumerate(downTextList)]
fig.show()
