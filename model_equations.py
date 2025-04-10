import numpy as np
import matplotlib.pyplot as plt

#defining script parameters:
region_length = 40

#defining parameters of the actual model
lambda_inh = 0.05
lambda_act = 0.3
k = 0.01
D_inh = 3.5 * 10e-10
D_c_inh = 3.5 * 10e-7
D_act = 6.9 * 10e-11
D_c_act = 5.9 * 10e-6
alpha = 0.1
h = 10
cm = 40

#"undefined" parameters / parameters that we do not have the values for yet
beta = 1
c0 = 1
n0 = 1

#we need 2 dimensional array because we only have n, c
def create_array(N:int): #not sure this condition still applies here?
    """
    Returns an initial condition 2D array with noise of length N.
    Parameters:
        N: the number of spatial points in the discretization
    """
    nc = np.ones((2, N)) #homogeneous stationary solution 
    nc = nc + np.random.uniform(0, 1, (2, N))/100 #1% amplitude additive noise
    return nc

nc = create_array(region_length)
    
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
    cinh_t = lap_c + fn_inh - lambda_inh * c
    cact_t = lap_c + fn_act - lambda_act * c

    if reaction == "activator":
        return (nact_t, cact_t)
    elif reaction == "inhibitor":
        return (ninh_t, cinh_t)
    else:
        raise ValueError("Invalid reaction value. Please use 'activator' or 'inhibitor' only.")

nact_t, cact_t = spatial_part(nc, reaction = "activator")
ninh_t, cinh_t = spatial_part(nc, reaction = "inhibitor")

def eulers_method_pde(dt:float = 0.01, reaction:str = "activator"):
    """
    Numerically integrates array nc obtained from spatial_part function using Explicit Euler's method.
    Parameters:
        dt: float specifying the time step for numerical integration.
    Returns a tuple of lists with 100 elements (frames) each.
    """
     
    narr_updates = []
    carr_updates = []

    for i in range(50000): 
        ut, vt = spatial_part(nc, reaction = reaction)
        #updating with explicit eulers method
        if i % 500 == 0: #appending every 500 iterations
            narr_updates.append(np.copy(nc[0]))
        nc[0] = nc[0] + ut * dt

        if i % 500 == 0:
            carr_updates.append(np.copy(nc[1]))
        nc[1] = nc[1] + vt * dt

        #boundary conditions:
        nc[:, 0] = nc[:, 1]
        nc[:, -1] = nc[:, -2]
    
    return (narr_updates, carr_updates)

narr_updates, carr_updates = eulers_method_pde()

def plot_static():
    """
    Creates a static plot of the last frame of animation of x versus c. 
    """
    #static plot:
    x_arr = np.linspace(0, region_length, region_length)
    print(f"x_arr = {x_arr.shape}")
    print(f"varr_updates[-1] = {carr_updates[-1].shape}")
    fig, ax = plt.subplots(1, 1)
    plt.plot(x_arr, carr_updates[-1])
    ax.set_xlim((0, region_length))
    ax.set_ylim((0, 5))
    plt.xlabel("x", fontsize = 15)
    plt.ylabel("v(x)", fontsize = 15)
    plt.title("Final frame for the concentration of mitosis regulating chemical")
    plt.show()

plot_static()


if __name__ == "__main__":
    pass