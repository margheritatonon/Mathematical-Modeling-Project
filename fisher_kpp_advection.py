import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    return n*(lambd - n) #not sure how to do the multiplication here, because n is an array and lambd is a value

def A(x, rho):
    """
    This is the expression inside of the integral
    """
    #we first take the neighbors of x that are in a radius of rho away

    return g()

if __name__ == "__main__":
    myx0 = wound(L=L)
    plot_initialcond = False
    if plot_initialcond == True:
        plt.plot(np.arange(-200, 201), myx0)
        plt.show()