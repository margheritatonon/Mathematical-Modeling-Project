import numpy as np

#defining parameters
lambda_inh = 0.05
lambda_act = 0.3
k = 0.01
D_inh = 3.5 * 10e-10
D_c_inh = 3.5 * 10e-7
D_act = 6.9 * 10e-11
D_c_act = 5.9 * 10e-6

#"undefined" parameters
h = 1
cm = 1
n = 1
beta = 1
c0 = 1
n0 = 1
alpha = 1




#we need 2 dimensional array because we only have n, c
def create_array(N:int):
    """
    Returns an initial condition 4D array with noise of length N.
    Parameters:
        N: the number of spatial points in the discretization
    """
    nc = np.ones((2, N)) #homogeneous stationary solution 
    nc = nc + np.random.uniform(0, 1, (2, N))/100 #1% amplitude additive noise
    return nc
    
def spatial_part(nc:np.array, dx:float = 1):
    """
    Implements a 1D finite difference numerical approximation to integrate the spatial part of the reaction-diffusion equations.
    Parameters:
        uv: a 2D array of initial conditions for u and v
        dx: the spatial step
    Returns:
        A tuple (ut, vt) of the PDEs
    """
    n, c = nc
    #computing laplacians - we are applying the 1D finite difference numerical scheme
    ninh_plus = np.roll(n, shift = dx)
    ninh_min = np.roll(n, shift = -dx)
    nact_plus = np.roll(n, shift = dx)
    nact_min = np.roll(n, shift = -dx)
    cinh_plus = np.roll(c, shift = dx)
    cinh_min = np.roll(c, shift = -dx)
    cact_plus = np.roll(c, shift = dx)
    cact_min = np.roll(c, shift = -dx)
    lap_ninh_plus = ninh_plus - 2*n + ninh_min
    lap_nact_plus = nact_plus - 2*n + nact_min
    lap_cinh_plus = cinh_plus - 2*c + cinh_min
    lap_cact_plus = cact_plus - 2*c + cact_min

    #the functions:
    sc_act = k * (((2*cm * (n-beta)*c) / (cm**2 + c**2)) + beta)
    sc_inh = ((h-1)*c + h * c0)
    fn_act = lambda_act * c0 + (n/n0) * ((n0**2 + alpha ** 2)/(n**2+alpha**2)) 
    fn_inh = lambda_inh * c0 * n / n0

    #the pdes:
    ninh_t = lap_
    ut = d1 * lap_u + gam * f
    vt = d2 * lap_v + gam * g

    return (ut, vt)