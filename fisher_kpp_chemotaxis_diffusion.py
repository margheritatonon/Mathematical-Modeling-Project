#this adds a diffusion term to the chemotaxis model.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fisher_kpp_chemotaxis import create_chemotaxis_array, animate_celldensity, animate_chemical, D, alpha, r, k, N

D_c = 0.4 #diffusion term for the chemical

def chemotaxis_eqs_diff(nc):
    """
    Sets up the Fisher-KPP model with chemotaxis for array nc, returning the PDEs dn/dt and dc/dt.
    """
    n, c = nc
    #computing laplacians - we are applying the 2D finite difference numerical scheme
    #moving up
    n_up = np.roll(n, shift=1, axis=0)
    c_up = np.roll(c, shift=1, axis=0)
    #moving down
    n_down = np.roll(n, shift=-1, axis=0)
    c_down = np.roll(c, shift = -1, axis = 0)
    #moving left
    n_left = np.roll(n, shift=1, axis=1)
    c_left = np.roll(c, shift=1, axis=1)
    #moving right
    n_right = np.roll(n, shift=-1, axis=1)
    c_right = np.roll(c, shift=-1, axis = 1)

    # 5 point stencil
    lap_n_5 = n_up + n_down + n_left + n_right - 4*n
    lap_c_5 = c_up + c_down + c_left + c_right - 4*c

    #equations:
    dndt = D * lap_n_5 - alpha * (n / np.clip(c, 1e-3, None)) * lap_c_5 + r * n * (1-n) #need to do + 1e-10 to avoid division by 0 errors that blow up the simulation
    dcdt = -k * n + D_c * lap_c_5

    return (dndt, dcdt)

def numerical_integration_explicit_eulers_diff(nc, dt = 0.01, num_iters = 50000):
    """
    Numerically integrates array nc obtained from chemotaxis_eqs function using Explicit Euler's method.
    """

    narr_updates = []
    carr_updates = []

    for i in range(num_iters): 
        nt, ct = chemotaxis_eqs_diff(nc) #recomputing the PDEs

        nc[0] = nc[0] + nt * dt #updating n
        nc[1] = nc[1] + ct * dt #updating c
        nc[1] = np.maximum(nc[1], 0) #prevent chemical from going negative

        #Boundary conditions:
        #dirichlet on vertical sides, there are always cells surrounding the wound: u = 1
        nc[0][:, 0] = 1
        nc[0][:, -1] = 1
        #neumann on horizontal sides, cell flux along the horizontal sides is none: u = u (this is especially important for when the wound vertically extends and "touches" the top and bottom of the grid)
        nc[0][0, :] = nc[0][1, :]
        nc[0][-1, :] = nc[0][-2, :]

        if i % 50 == 0: #appending every 50 iterations
            narr_updates.append(np.copy(nc[0]))
            carr_updates.append(np.copy(nc[1]))
    
    return narr_updates, carr_updates


if __name__ == "__main__":
    nc = create_chemotaxis_array(N, shape = "oval")
    nt, ct = chemotaxis_eqs_diff(nc)
    narr_updates, carr_updates = numerical_integration_explicit_eulers_diff(nc)
    #animate_celldensity(narr_updates, N)
    animate_chemical(carr_updates, N)