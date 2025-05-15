import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import matplotlib.gridspec as gridspec


alpha = 0.2 #chemotaxis parameter
k = 0.01 #degradation rate of the chemotactic signal
D = 0.2 
r = 0.1 
N  = 100 
D2 = 0.15

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

def chemotaxis_eqs(nc):
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
    dcdt = -k * n + D2 * lap_c_5

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

def animate_celldensity(narr_updates, N):
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
    plt.title(f"2D Fisher-KPP Model with Chemotaxis: Cell Density Animation", fontsize=19)
    plt.show()

def animate_chemical(carr_updates, N):
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
    plt.title(f"2D Fisher-KPP Model with Chemotaxis: Chemical Concntration Animation", fontsize=19)
    plt.show()

def plot_static_snapshots_density(uarr_updates, N, times, dt):
    """
    Plots static snapshots at specified times of the uarr_updates array.
    """
    iterations = [int(t / dt) for t in times]
    frame_indices = [i // 50 for i in iterations]

    num_snapshots = len(times)
    ncols = 4
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

    fig.suptitle(f"Cell Density\nD = {D}, Dc = {D2}, α = {alpha}, r = {r}, κ = {k}", fontsize=25)
    plt.subplots_adjust(top=0.8, bottom=0.08)
    plt.tight_layout()
    plt.show()

def plot_static_snapshots_chemical(carr_updates, N, times, dt):
    """
    Plots static snapshots at specified times of the carr_updates array.
    """
    iterations = [int(t / dt) for t in times]
    frame_indices = [i // 50 for i in iterations]

    num_snapshots = len(times)
    ncols = 4
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

    fig.suptitle(f"Chemical Concentration\nD = {D}, Dc = {D2}, α = {alpha}, r = {r}, κ = {k}", fontsize=25)
    plt.subplots_adjust(top=0.8, bottom=0.08)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nc = create_chemotaxis_array(N, shape = "oval")
    nt, ct = chemotaxis_eqs(nc)
    narr_updates, carr_updates = numerical_integration_explicit_eulers(nc)
    animate_celldensity(narr_updates, N)
    animate_chemical(carr_updates, N)
    plot_static_snapshots_density(narr_updates, N, [0, 50, 100, 150, 200, 250, 300, 350], dt = 0.01)
    plot_static_snapshots_chemical(carr_updates, N, [0, 50, 100, 150, 200, 250, 300, 350], dt = 0.01)

