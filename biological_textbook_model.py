import numpy as np

#defining parameters
lambda_inh = 0.05
lambda_act = 0.3
k = 0.01
D_inh = 3.5 * 10e-10
D_c_inh = 3.5 * 10e-7
D_act = 6.9 * 10e-11
D_c_act = 5.9 * 10e-6

#"undefined" parameters / parameters that we do not have the values for yet
h = 1
cm = 1
n = 1
beta = 1
c0 = 1
n0 = 1
alpha = 1
lam = 1




#we need 2 dimensional array because we only have n, c
def create_array(N:int): #not sure this condition still applies here?
    """
    Returns an initial condition 4D array with noise of length N.
    Parameters:
        N: the number of spatial points in the discretization
    """
    nc = np.ones((2, N)) #homogeneous stationary solution 
    nc = nc + np.random.uniform(0, 1, (2, N))/100 #1% amplitude additive noise
    return nc
    
def spatial_part(nc:np.array, dx:float = 1, reaction:str = "activator"):
    """
    Implements a 1D finite difference numerical approximation to integrate the spatial part of the reaction-diffusion equations.
    Parameters:
        nc: a 2D array of initial conditions for n and c
        dx: the spatial step
        reaction: either activator or inhibitor
    Returns:
        A tuple (ut, vt) of the PDEs
    """
    n, c = nc
    #computing laplacians - we are applying the 1D finite difference numerical scheme
    n_plus = np.roll(n, shift = dx)
    n_min = np.roll(n, shift = -dx)
    c_plus = np.roll(c, shift = dx)
    c_min = np.roll(c, shift = -dx)
    lap_n = n_plus - 2*n + n_min
    lap_c = c_plus - 2*c + c_min

    #the functions:
    sc_act = k * (((2*cm * (n-beta)*c) / (cm**2 + c**2)) + beta)
    sc_inh = ((h-1)*c + h * c0)
    fn_act = lambda_act * c0 + (n/n0) * ((n0**2 + alpha ** 2)/(n**2+alpha**2)) 
    fn_inh = lambda_inh * c0 * n / n0

    #the pdes:
    ninh_t = lap_n + sc_inh * n * (2-n/n0) - k*n
    nact_t = lap_n + sc_act * n * (2-n/n0) - k*n
    cinh_t = lap_c + fn_inh - lam * c
    cact_t = lap_c + fn_act - lam * c

    if reaction == "activator":
        return (nact_t, cact_t)
    elif reaction == "inhibitor":
        return (ninh_t, cinh_t)
    else:
        raise ValueError("Invalid reaction value. Please use 'activator' or 'inhibitor' only.")

