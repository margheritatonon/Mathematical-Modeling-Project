import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import integrate

#defining parameters:
lambd = 7 
alpha = 0.1 
rho = 16.9 #the radius for nonlocal integration
L = 200 #size of domain --> but halved (domain goes from -L to L)

#clarifying notation:
#when we perform the integral, we are integrating with respect to xhat for one specific value of x.
#xhat are basically the x values that are in a certain radius away from x.
#xhat is a displacement vector representing the relative position between two interacting cells
#therefore for all x values we need to compute this integral (i think)
#the integral is a nonlocal integral that aggregates information about the cell density n in a neighborhood of x ([x−rho,x+rho])
#so when we have n(x+xhat, t) in the integral, we are basically just integrating over a neighboring region.



#h(x) does not depend on time so we can calculate those values once and use them throughtout.


#we begin by discretizing the domain and defining the initial wound
#in the paper, the domain is 1D.
def wound(L=L):
    """
    This creates the -L to L initial condition array of the wound.
    This actually represents a cross section of the wound, viewed laterally.
    """
    x = np.arange(-L, L+1)
    x0 = 1.0 - 0.75 * np.exp(-(0.1 * x) ** 6) #the initial condition they define in the paper
    return x0


#defining a function as the expression inside the integral:
def g(n):
    """
    Defines the function g for the array of n values.
    """
    return n * (lambd - n) 

def h(xhat):
    """
    Defines the h(x) function for the array of values that neighbor the
    Parameters:
        xhat: the indices of the neighboring values of a certain value x.
    """
    return (0.1*np.arctan(0.2*xhat))/(np.arctan(2))

def A(narr, currx, rho):
    """
    This is the expression inside of the integral.
    Parameters:
        narr: the array of current n values
        currx: the x index we look at
        rho: the radius we consider
    """
    #we first take the neighbors of x that are in a radius of rho away
    #we therefore apply a mask
    #the thing is that its one dimensional so we just take the neighbors of x + rho, x-rho
    indices = np.arange(min(0, currx-rho), min(currx+rho, len(narr))) #prevents errors in indexing, in case currx-rho < 0 or currx+rho > the total array length
    #now we index n with these indices
    ns = narr[indices]
    a = g(ns) * h(indices) #this is the expression that is inside of the integral
    return a  #we will need to integrate a numerically, from -rho to rho.

def integrating_expression(to_integrate, rho):
    """
    Integrates the to_integrate array using the trapezoidal rule.
    """
    x = np.linspace(-rho, rho, len(to_integrate)) #the integration bounds
    result = integrate.trapezoid(to_integrate, x)
    return result

def f(n):
    """
    Defining the f(n) function in the expression.
    """
    return n*(1-n)

def before_singlepartial(integrated, n):
    """
    Defines the expression before the partial derivative with respect to x is taken.
    """
    return n * integrated

def partial_wrt_x(expression, dx = 1):
    """
    Computing the partial derivative of the expression n*the integral in the formula.
    We use central differences to approximate the derivative.
    """
    derivative = np.zeros_like(expression)
    derivative[1:-1] = (expression[2:] - expression[:-2]) / (2 * dx)
    derivative[0] = (expression[1] - expression[0]) / dx
    derivative[-1] = (expression[-1] - expression[-2]) / dx

    return derivative


if __name__ == "__main__":
    myx0 = wound(L=L)
    plot_initialcond = False
    if plot_initialcond == True:
        plt.plot(np.arange(-200, 201), myx0)
        plt.show()