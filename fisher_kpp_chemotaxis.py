import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fisher_kpp import  N
import math
import matplotlib.gridspec as gridspec
import os

D = 0.05
r = 0.1
alpha = 0.1 #chemotaxis parameter
k = 0.05 #degradation rate of the chemotactic signal
dx = 1

def create_chemotaxis_array(N:int, shape:str = "circle"):
    """
    Returns an initial condition 2D array.
    Parameters:
        N: the number of spatial points in the discretization
        shape: the shape of the initial condition (e.g., "circle", "rectangle", "oval", "elongated rectangle")
    """

    nc = np.zeros((2, N, N))
    nc[0] = 1  # cell density everywhere
    nc[1] = 0  # chemical concentration everywhere

    center = N // 2
    Y, X = np.ogrid[:N, :N]

    if shape == "circle":
        radius = N // 5
        distance = np.sqrt((X - center)**2 + (Y - center)**2)
        mask = distance <= radius

    elif shape == "rectangle":
        mask = np.zeros((N, N), dtype=bool)
        mask[N//4:N//2, N//4:N//2] = True

    elif shape == "oval":
        height = N // 4 
        width = N // 6       
        mask = ((X - center)**2) / (width**2) + ((Y - center)**2) / (height**2) <= 1

    elif shape == "elongated rectangle":
        mask = np.zeros((N, N), dtype=bool)
        mask[N//3:2*N//3, N//4:3*N//4] = True

    else:
        raise ValueError(f"Shape '{shape}' not recognized.")

    # Apply wound: remove cells, add chemical
    nc[0][mask] = 0  # cells removed in wound
    nc[1][mask] = 1  # chemical only in wound

    return nc

def chemotaxis_eqs(nc, dx = dx):
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

    #computing the chemotaxis part of the equation
    c_safe = np.clip(c, 1e-3, None) #we need to make sure we don't divide by 0 in the n/c term, we want to avoid instability
    grad_c_y, grad_c_x = np.gradient(c, dx) #spatial gradients of c
    grad_nc_y, grad_nc_x = np.gradient(n / c_safe, dx) #spatial gradients of n/c
    grad_term = grad_nc_y * grad_c_y + grad_nc_x * grad_c_x #dot product of terms
    chemotaxis_term = grad_term + (n / c_safe) * lap_c_5

    #equations:
    dndt = dndt = D * lap_n_5 - alpha * chemotaxis_term + r * n * (1 - n) 
    dcdt = -k * n

    return (dndt, dcdt)

def numerical_integration_explicit_eulers(nc, dt = 0.01, num_iters = 50000):
    """
    Numerically integrates array nc obtained from chemotaxis_eqs function using Explicit Euler's method.
    """

    narr_updates = []
    carr_updates = []

    for i in range(num_iters): 
        nt, ct = chemotaxis_eqs(nc) #recomputing the PDEs

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

def animate_celldensity(narr_updates, N, save_path = None, showing = True):
    """
    Creates an animation of the cell density over time.
    """
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
    	narr_updates[0],
    	interpolation="bilinear",
    	vmin=0,
    	vmax=1,
    	origin="lower",
    	extent=[0, N, 0, N],
	)

    def update(frame):
        im.set_array(narr_updates[frame])
        #im.set_clim(vmin=np.min(narr_updates[frame]), vmax=np.max(narr_updates[frame]) + 0.01)
        return (im, )
    

    ani = animation.FuncAnimation(
    	fig, update, interval=50, blit=True, frames = len(narr_updates), repeat = False
	)
    plt.title(f"Cell Density at α = {alpha}, κ = {k}", fontsize=19)

    if save_path:
        if os.path.exists(save_path):
            raise FileExistsError(f"File '{save_path}' already exists. Unable to save animation. Change path name or delete old file.")
        ani.save(save_path, writer="ffmpeg", fps=20)
        print(f"Animation saved to {save_path}")
        plt.show()
    else:
        plt.show()

    if showing:
        plt.show()

def animate_chemical(carr_updates, N, save_path = None, showing = True):
    """
    Creates an animation of the chemical concentration over time.
    """
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
    	carr_updates[0],
    	interpolation="bilinear",
    	vmin=0,
    	vmax=1,
    	origin="lower",
    	extent=[0, N, 0, N],
	)

    def update(frame):
        im.set_array(carr_updates[frame])
        #im.set_clim(vmin=np.min(carr_updates[frame]), vmax=np.max(carr_updates[frame]) + 0.01)
        return (im, )
    

    ani = animation.FuncAnimation(
    	fig, update, interval=50, blit=True, frames = len(carr_updates), repeat = False
	)
    plt.title(f"Chemical Concentration at α = {alpha}, κ = {k}", fontsize=19)

    if save_path:
        if os.path.exists(save_path):
            raise FileExistsError(f"File '{save_path}' already exists. Unable to save animation. Change path name or delete old file.")
        ani.save(save_path, writer="ffmpeg", fps=20)
        print(f"Animation saved to {save_path}")
        plt.show()
    else:
        plt.show()
    
    if showing:
        plt.show()

def plot_static_snapshots_density(uarr_updates, N, times, dt):
    """
    Plots static snapshots at specified times of the uarr_updates array.
    """
    iterations = [int(t / dt) for t in times]
    frame_indices = [i // 50 for i in iterations]

    num_snapshots = len(times)
    ncols = 3
    nrows = math.ceil(num_snapshots / ncols)

    # Create figure with gridspec to make room for left-side colorbar
    fig = plt.figure(figsize=(3.5 * ncols, 3 * nrows))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[0.1] + [1]*ncols, wspace=0.3)

    axes = []
    for i in range(num_snapshots):
        row = i // ncols
        col = i % ncols + 1  # Shift by 1 because column 0 is for colorbar
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

    for ax, idx, t in zip(axes, frame_indices, times):
        im = ax.imshow(
            uarr_updates[idx],
            cmap='viridis',
            origin='lower',
            extent=[0, N, 0, N],
            vmin=0,
            vmax=1
        )
        ax.set_title(f"t = {t}", fontsize=12)
        ax.axis('off')

    cax = fig.add_subplot(gs[:, 0])
    fig.colorbar(im, cax=cax)
    #cax.set_ylabel('Concentration', rotation=270, labelpad=15)

    fig.suptitle(f"Cell Density at α = {alpha}, κ = {k}", fontsize=25)
    plt.subplots_adjust(top=0.85, bottom=0.08)
    plt.tight_layout()
    plt.show()

def plot_static_snapshots_chemical(carr_updates, N, times, dt):
    """
    Plots static snapshots at specified times of the carr_updates array.
    """
    iterations = [int(t / dt) for t in times]
    frame_indices = [i // 50 for i in iterations]

    num_snapshots = len(times)
    ncols = 3
    nrows = math.ceil(num_snapshots / ncols)

    # Create figure with gridspec to make room for left-side colorbar
    fig = plt.figure(figsize=(3.5 * ncols, 3 * nrows))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[0.1] + [1]*ncols, wspace=0.3)

    axes = []
    for i in range(num_snapshots):
        row = i // ncols
        col = i % ncols + 1  # Shift by 1 because column 0 is for colorbar
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

    for ax, idx, t in zip(axes, frame_indices, times):
        im = ax.imshow(
            carr_updates[idx],
            cmap='viridis',
            origin='lower',
            extent=[0, N, 0, N],
            vmin=0,
            vmax=1
        )
        ax.set_title(f"t = {t}", fontsize=12)
        ax.axis('off')

    cax = fig.add_subplot(gs[:, 0])
    fig.colorbar(im, cax=cax)
    #cax.set_ylabel('Concentration', rotation=270, labelpad=15)

    fig.suptitle(f"Chemical Concentration at α = {alpha}, κ = {k}", fontsize=25)
    plt.subplots_adjust(top=0.85, bottom=0.08)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nc = create_chemotaxis_array(N, shape = "rectangle")
    nt, ct = chemotaxis_eqs(nc)
    narr_updates, carr_updates = numerical_integration_explicit_eulers(nc)

    save_path_density = "animations/chemotaxis_density_rectangular_a01_k005.gif"
    save_path_chemical = "animations/chemotaxis_chemical_a01_k005.gif"
    animate_celldensity(narr_updates, N, save_path=save_path_density, showing=True)
    #animate_chemical(carr_updates, N, save_path=save_path_chemical, showing = False)

    #plot_static_snapshots_density(narr_updates, N, [1, 10, 50, 100, 150, 200], dt = 0.01)
    #plot_static_snapshots_chemical(carr_updates, N, [0, 10, 50, 100, 150, 200], dt = 0.01)

