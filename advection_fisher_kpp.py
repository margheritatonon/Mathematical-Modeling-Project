import numpy as np

#defining the parameters
lambd = 7 #g parameter
alpha = 1 #parameter for diffusion
rho = 8 #the radius for nonlocal integration
L = 200 #size of domain --> but halved (because domain goes from -L to L)
dx = 0.25 #spatial integration/derivative step
dt = 0.01 #time step

#VARIABLE DESCRIPTION:
#in the equations:
    # n represents the cell density, t is the time, and x is the spatial position. We are in 1 dimension so x is just one dimensional (not a vector).

#in the code:
    #xhat is INDICES of the range [x-rho, x+rho]. You have a different xhat array depending on which value of x you look at, as the array needs to be computed for every x in the discretization.
    #narr is the array of cell densities. it has length 2L. the values inside must range from 0 to 1.


#INITIAL CONDITION
def wound(L=L, dx = dx):
    """
    This creates the -L to L initial condition array of the wound.
    This actually represents a cross section of the wound, viewed laterally.
    """
    x = np.arange(-L, L+dx, dx)
    x0 = 1.0 - 0.75 * np.exp(-(0.1 * x) ** 6) #the initial condition they define in the paper
    return x0


#We can divide the PDE into 3 blocks: the Laplacian, the reaction term, and the advection integral term.

#REACTION TERM:
def f(narr:np.array):
    """
    Computes the reaction part of the PDE.
    The output will be a 1D array of length 2L.
    """
    return narr*(1-narr)

#LAPLACIAN
def laplacians(narr, dx=dx):
    """
    Defines the laplacian (second partial derivative wrt x) of narr, the cell density array.
    This is a 1D Laplacian (we therefore use a three point stencil).
    The output is also an array of length 2L.
    """
    lap = np.zeros_like(narr)
    lap[1:-1] = (narr[2:] - 2 * narr[1:-1] + narr[:-2]) / dx**2
    #neumann boundary conditions (zero flux)
    lap[0] = lap[1]
    lap[-1] = lap[-2]
    return lap


#INTEGRAL!!