import numpy as np

#defining the parameters
lambd = 7 #g parameter
alpha = 1 #parameter for diffusion
rho = 8 #the radius for nonlocal integration
L = 200 #size of domain --> but halved (because domain goes from -L to L)
dx = 0.5 #spatial integration/derivative step
dt = 0.01 #time step

#VARIABLE DESCRIPTION:
#in the equations:
    # n represents the cell density, t is the time, and x is the spatial position. We are in 1 dimension so x is just one dimensional (not a vector).

#in the code:
    #xhat is INDICES of the range [x-rho, x+rho]. You have a different xhat array depending on which value of x you look at, as the array needs to be computed for every x in the discretization.
    #narr is the array of cell densities. the values inside must range from 0 to 1.
        #the length of the narr array depends on dx. 
            #if dx = 1, the length is 2L. if dx = 0.5, the length is 2*2L. if dx = 0.25, the length is 4*2L, etc.


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
    This is a 1D Laplacian. We use a finite difference approximation.
    The output is also an array of length 2L.
    """
    left = np.roll(narr, 1) #this is 1 and -1, but each index moves dx units in the spatial domain because of how the narr array is constructed. therefore we are shifting by dx
    right = np.roll(narr, -1)
    lap_u = (left - 2*narr + right) / (dx**2) #finite difference formula.
    return lap_u


#INTEGRAL!!
#for every x in the array, we need to compute this integral as it is a nonlocal integral that looks at different values of x.
    #we need to adjust the rho value accordingly (rho assumes dx = 1) --> radius = int(rho / dx)

def integral_term(narr, dx=dx, rho = rho):
    """
    Computes the expression inside of the integral term.
    """
    radius = int(rho/dx) #we access the radius like this as we need to take into account the value of dx as well.

    #we now can compute the "kernel" h. h depends only on xhat, which is just an offset which depends on rho. it does NOT depend on x+- rho.
    #we therefore compute h outside of the for loop.
    #this is the RIGHT part of the integral.
    j = np.arange(-radius, radius + 1)
    xhat = j*dx #this converts to physical distances.
    h_kernel = 0.1 * np.arctan(0.2 * xhat) / np.arctan(2.0)

    #for every x in the array, we need to compute this.
    #begin with a for loop implementation
    integrand_values = [] #this should be a 2L/dx length array.
    for i in range(len(narr)):
        #LEFT part of the integral
        n_neighbors = narr[max(i - radius, 0) : min(i + radius + 1, len(narr))] #we account for the boundaries with the min and max terms.
        #n_neighbors is all of the neighbors that are rho away from the current x that we look at.
        g_function = n_neighbors * (lambd - n_neighbors) #this is one part of the 

        #putting together the integral:
        integrand = g_function * h_kernel 
        integrand_values.append(integrand)
    
    return np.array(integrand_values)
        


        


