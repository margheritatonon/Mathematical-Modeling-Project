import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import matplotlib.gridspec as gridspec
import os #to check if animation already exists

D = 0.05 #according to the paper, D can range anywhere from 10^-8 to 0.15 --> 0.05
r = 0.1 #according to the paper, r can range anywhere from 10^-6 to 0.5 --> 0.1
N  = 100 

def create_array(N:int, shape:str = "circle"):
    """
    Returns an initial condition 2D array.
    Parameters:
        N: the number of spatial points in the discretization
        shape: the shape of the initial condition (e.g., "circle", "rectangle", "oval", "elongated rectangle")
    """

    #ones, as we are assuming the monolayer is at confluence at t=0
    uv = np.ones((2, N, N)) #2D array

    #we can simulate different initial cell patterns here (different shapes of the zone of 0s)
    #circle:
    if shape == "circle":
        center = N // 2
        radius = N // 5
        Y, X = np.ogrid[:N, :N]
        distance = np.sqrt((X - center)**2 + (Y - center)**2)
        mask = distance <= radius
        #set both layers (u and v) to 0 inside the wound
        uv[0][mask] = 0
        uv[1][mask] = 0
        return uv

    #rectangle:
    if shape == "rectangle":
        uv[0][N//4:N//2, N//4:N//2] = 0
        uv[1][N//4:N//2, N//4:N//2] = 0
        return uv

    #elongated rectangle:
    if shape == "elongated rectangle":
        uv[0][:, N//4:N//2] = 0
        uv[1][:, N//4:N//2] = 0
        return uv

    #oval:
    if shape == "oval":
        center = N // 2
        height = N // 4 
        width = N // 6       
        Y, X = np.ogrid[:N, :N]
        mask = ((X - center)**2) / (width**2) + ((Y - center)**2) / (height**2) <= 1

        uv[0][mask] = 0
        uv[1][mask] = 0
        return uv

def fisher_kpp(uv, r=r, D=D):
    """
    Sets up the Fisher-KPP model for array uv, returning the PDEs du/dt and dv/dt.
    """
    u, v = uv
    #computing laplacians - we are applying the 2D finite difference numerical scheme
    #moving up
    u_up = np.roll(u, shift=1, axis=0)
    v_up = np.roll(v, shift=1, axis=0)
    #moving down
    u_down = np.roll(u, shift=-1, axis=0)
    v_down = np.roll(v, shift = -1, axis = 0)
    #moving left
    u_left = np.roll(u, shift=1, axis=1)
    v_left = np.roll(v, shift=1, axis=1)
    #moving right
    u_right = np.roll(u, shift=-1, axis=1)
    v_right = np.roll(v, shift=-1, axis = 1)

    # 5 point stencil
    lap_u_5 = u_up + u_down + u_left + u_right - 4*u 
    lap_v_5 = v_up + v_down + v_left + v_right - 4*v

    #the pdes:
    ut = D * lap_u_5 + r * u * (1-u)
    vt = D * lap_v_5 + r * v * (1-v)

    return (ut, vt)

def numerical_integration_explicit_eulers(uv:np.array, dt:float=0.01, num_iters:int=50000):
    """
    Numerically integrates array uv obtained from fisher_kpp function using Explicit Euler's method.
    """

    uarr_updates = []
    varr_updates = []

    for i in range(num_iters): 
        ut, vt = fisher_kpp(uv) #recomputing the PDEs

        uv[0] = uv[0] + ut * dt #updating u 
        uv[1] = uv[1] + vt * dt #updating v

        #Boundary conditions:
        #Neumann on everything
        uv[0][0, :] = uv[0][1, :]       
        uv[0][-1, :] = uv[0][-2, :]    
        uv[0][:, 0] = uv[0][:, 1]   
        uv[0][:, -1] = uv[0][:, -2] 
        uv[1][0, :] = uv[1][1, :]       
        uv[1][-1, :] = uv[1][-2, :] 
        uv[1][:, 0] = uv[1][:, 1] 
        uv[1][:, -1] = uv[1][:, -2]

        if i % 50 == 0: #appending every 50 iterations
            uarr_updates.append(np.copy(uv[0]))
            varr_updates.append(np.copy(uv[1]))
    
    return uarr_updates, varr_updates

def animate_plot(single_integrated_array, N, save_path = None):
    """
    Animates the plot of the numerically integrated solution.
    Single_integrated_array is the array of u or v values, after numerical integration.
    """
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
    	single_integrated_array[0],
    	interpolation="bilinear",
    	vmin=0,
    	vmax=1,
    	origin="lower",
    	extent=[0, N, 0, N],
	)

    def update(frame):
        im.set_array(single_integrated_array[frame])
        #im.set_clim(vmin=np.min(single_integrated_array[frame]), vmax=np.max(single_integrated_array[frame]) + 0.01)
        return (im, )
    

    ani = animation.FuncAnimation(
    	fig, update, interval=10, blit=True, frames = len(single_integrated_array), repeat = False
	)
    plt.title(f"$r = {r}, D = {D}$", fontsize=19)

    if save_path:
        if os.path.exists(save_path):
            raise FileExistsError(f"File '{save_path}' already exists. Unable to save animation. Change path name or delete old file.")
        ani.save(save_path, writer="ffmpeg", fps=20)
        print(f"Animation saved to {save_path}")
        plt.show()
    else:
        plt.show()


def plot_static_snapshots(uarr_updates, N, times, dt):
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

    fig.suptitle(f"Snapshots of 2D Fisher-KPP at r = {r}, D = {D}", fontsize=25)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    uv = create_array(N, shape = "oval")
    #plt.imshow(uv[0], cmap='gray', origin='lower')
    #plt.show()

    ut, vt = fisher_kpp(uv)

    uarr_updates, varr_updates = numerical_integration_explicit_eulers(uv)

    save_path = f"animations/fkpp_animation_r10e5_D005.gif"
    #animate_plot(uarr_updates, N, save_path=save_path)
    #animate_plot(uarr_updates, N)

    plot_static_snapshots(uarr_updates, N, dt=0.01, times = [1, 10, 50, 100, 150, 200])
    