# Topic      : Bisection Method - Roots of Equations
# Simulation : Graphical Simulation of the Method
# Language   : Python v3.8.3
# Authors    : Nathan Collier, Autar Kaw
# Translate  : Furkan Kıyıkçı
# Date       : 6 September 2002
# Abstract   : This simulation shows how the bisection method works for finding roots of
#              an equation f(x)=0
#


import numpy as np
import matplotlib.pyplot as plt

# INPUTS: Enter the following
# Function in f(x)=0
f = lambda x: (x ** 3) - 0.165 * (x ** 2) + 3.993 * (10 ** (-4))

# Upper initial guess
xu = 0.11
# Lower initial guess
xl = 0.0
# Lower bound of range of 'x' to be seen
lxrange = -0.02
# Upper bound of range of 'x' to be seen
uxrange = 0.12
#
# SOLUTION

# The following finds the upper and lower 'y' limits for the plot based on the given
#    'x' range in the input section.
maxi = f(lxrange)
mini = f(lxrange)

for i in np.arange(lxrange, uxrange, (uxrange - lxrange) / 10):
    if f(i) > maxi:
        maxi = f(i)
    if f(i) < mini:
        mini = f(i)

tot = maxi - mini
mini = mini - 0.1 * tot
maxi = maxi + 0.1 * tot

x = np.arange(lxrange, uxrange, 0.0000001)
y = f(x)

# This graphs the function and two lines representing the two guesses

fig, ax = plt.subplots(figsize=(15, 10))  # Create a figure containing a single axes.

ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.
ax.plot(x, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([xl, xl], [maxi, mini], 'y')
ax.plot([xu, xu], [maxi, mini], 'g')
ax.plot([lxrange, uxrange], [0, 0], 'k')
fig.show()
# --------------------------------------------------------------------------------
# Iteration 1
xr = (xu + xl) / 2
# This graphs the function and two lines representing the two guesses
fig = plt.figure(figsize=(15, 10))  # Create a figure containing a single axes.
ax = plt.subplot(312)
ax.plot(x, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([xl, xl], [maxi, mini], 'y')
ax.plot([xu, xu], [maxi, mini], 'g')
ax.plot([xr, xr], [maxi, mini], 'r')
ax.plot([lxrange, uxrange], [0, 0], 'k')
ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.

# This portion adds the text and math to the top part of the figure window
upTextlist = [
    'Iteration 1',
    'xr = (xu + xl)/2 = {}'.format(str(xr)),
    'Finding the value of the function at the lower and upper guesses and the estimated root',
    'f(xl) = {}'.format(str(f(xl))),
    'f(xu) = {}'.format(str(f(xu))),
    'f(xr) = {}'.format(str(f(xr)))
]
[plt.figtext(0.1, .9 - i * .03, text) for i, text in enumerate(upTextlist)]

# Check the interval between which the root lies

if f(xu) * f(xr) < 0:
    xl = xr
else:
    xu = xr
# This portion adds the text and math to the bottom part of the figure window

downTextlist = [
    'Check the interval between which the root lies. Does it lie in ( xl , xu ) or ( xr , xu )?',
    'xu = {}'.format(xu),
    'xl = {}'.format(xl)]
[plt.figtext(0.1, .3 - i * .03, text) for i, text in enumerate(downTextlist)]
fig.show()
xp = xr
# --------------------------------------------------------------------------------
# Iteration 2
xr = (xu + xl) / 2
fig = plt.figure(figsize=(15, 10))  # Create a figure containing a single axes.
ax = plt.subplot(312)
ax.plot(x, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([xl, xl], [maxi, mini], 'y')
ax.plot([xu, xu], [maxi, mini], 'g')
ax.plot([xr, xr], [maxi, mini], 'r')
ax.plot([lxrange, uxrange], [0, 0], 'k')
ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.
upTextList = [
    'Iteration 2',
    'xr = (xu + xl) / 2 = {}'.format(xr),
    'Finding the value of the function at the lower and upper guesses and the estimated root',
    'f(xl) = {}'.format(f(xl)),
    'f(xu) = {}'.format(f(xu)),
    'f(xr) = {}'.format(f(xr))
]
[plt.figtext(0.1, .9 - i * .03, text) for i, text in enumerate(upTextList)]

if f(xu) * f(xr) < 0:
    xl = xr
else:
    xu = xr

ea = abs((xr - xp) / xr) * 100
# This portion adds the text and math to the bottom part of the figure window

downTextList = [
    'Absolute relative approximate error',
    'ea = abs((xr - xp) / xr)*100 = {}%'.format(ea),
    'Check the interval between which the root lies. Does it lie in ( xl , xu ) or ( xr , xu )?',
    'xu = {}'.format(xu),
    'xl = {}'.format(xl)
]
[plt.figtext(0.1, .3 - i * .03, text) for i, text in enumerate(downTextList)]
fig.show()
xp = xr
# --------------------------------------------------------------------------------
# Iteration 3
fig = plt.figure(figsize=(15, 10))  # Create a figure containing a single axes.
ax = plt.subplot(312)
xr = (xu + xl) / 2
ax.plot(x, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([xl, xl], [maxi, mini], 'y')
ax.plot([xu, xu], [maxi, mini], 'g')
ax.plot([xr, xr], [maxi, mini], 'r')
ax.plot([lxrange, uxrange], [0, 0], 'k')
ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.
upTextList = [
    'Iteration 3',
    'xr = (xu + xl) / 2 = {}'.format(xr),
    'Finding the value of the function at the lower and upper guesses and the estimated root',
    'f(xl) = {}'.format(f(xl)),
    'f(xu) = {}'.format(f(xu)),
    'f(xr) = {}'.format(f(xr))
]
[plt.figtext(0.1, .9 - i * .03, text) for i, text in enumerate(upTextList)]

if f(xu) * f(xr) < 0:
    xl = xr
else:
    xu = xr

ea = abs((xr - xp) / xr) * 100

downTextList = [
    'Absolute relative approximate error',
    'ea = abs((xr - xp) / xr)*100 = {}%'.format(ea),
    'Check the interval between which the root lies. Does it lie in ( xl , xu ) or ( xr , xu )?',
    'xu = {}'.format(xu),
    'xl = {}'.format(xl)
]
[plt.figtext(0.1, .3 - i * .03, text) for i, text in enumerate(downTextList)]
fig.show()
xp = xr
# --------------------------------------------------------------------------------
# Iteration 4
fig = plt.figure(figsize=(15, 10))  # Create a figure containing a single axes.
ax = plt.subplot(312)
xr = (xu + xl) / 2
ax.plot(x, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([xl, xl], [maxi, mini], 'y')
ax.plot([xu, xu], [maxi, mini], 'g')
ax.plot([xr, xr], [maxi, mini], 'r')
ax.plot([lxrange, uxrange], [0, 0], 'k')
ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.
upTextList = [
    'Iteration 4',
    'xr = (xu + xl) / 2 = {}'.format(xr),
    'Finding the value of the function at the lower and upper guesses and the estimated root',
    'f(xl) = {}'.format(f(xl)),
    'f(xu) = {}'.format(f(xu)),
    'f(xr) = {}'.format(f(xr))
]
[plt.figtext(0.1, .9 - i * .03, text) for i, text in enumerate(upTextList)]

if f(xu) * f(xr) < 0:
    xl = xr
else:
    xu = xr

ea = abs((xr - xp) / xr) * 100

downTextList = [
    'Absolute relative approximate error',
    'ea = abs((xr - xp) / xr)*100 = {}%'.format(ea),
    'Check the interval between which the root lies. Does it lie in ( xl , xu ) or ( xr , xu )?',
    'xu = {}'.format(xu),
    'xl = {}'.format(xl)
]
[plt.figtext(0.1, .3 - i * .03, text) for i, text in enumerate(downTextList)]
fig.show()
xp = xr
# --------------------------------------------------------------------------------
# Iteration 5
fig = plt.figure(figsize=(15, 10))  # Create a figure containing a single axes.
ax = plt.subplot(312)
xr = (xu + xl) / 2
ax.plot(x, y, label='x^3-0.165*x^2+3.993*10^-4')
ax.legend(loc=1)  # Add a legend.
ax.plot([xl, xl], [maxi, mini], 'y')
ax.plot([xu, xu], [maxi, mini], 'g')
ax.plot([xr, xr], [maxi, mini], 'r')
ax.plot([lxrange, uxrange], [0, 0], 'k')
ax.set_title("Entered function on given interval with initial guesses")  # Add a title to the axes.
upTextList = [
    'Iteration 5',
    'xr = (xu + xl) / 2 = {}'.format(xr),
    'Finding the value of the function at the lower and upper guesses and the estimated root',
    'f(xl) = {}'.format(f(xl)),
    'f(xu) = {}'.format(f(xu)),
    'f(xr) = {}'.format(f(xr))
]
[plt.figtext(0.1, .9 - i * .03, text) for i, text in enumerate(upTextList)]

if f(xu) * f(xr) < 0:
    xl = xr
else:
    xu = xr

ea = abs((xr - xp) / xr) * 100

downTextList = [
    'Absolute relative approximate error',
    'ea = abs((xr - xp) / xr)*100 = {}%'.format(ea),
    'Check the interval between which the root lies. Does it lie in ( xl , xu ) or ( xr , xu )?',
    'xu = {}'.format(xu),
    'xl = {}'.format(xl)
]
[plt.figtext(0.1, .3 - i * .03, text) for i, text in enumerate(downTextList)]
fig.show()
xp = xr
