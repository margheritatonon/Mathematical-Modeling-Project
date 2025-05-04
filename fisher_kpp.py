import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

D = 0.05 #according to the paper, D can range anywhere from 10e-8 to 0.15
r = 0.1 #according to the paper, r can range anywhere from 10e-6 to 0.5
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

def fisher_kpp(uv):
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
        #dirichlet on vertical sides, there are always cells surrounding the wound: u = 1
        uv[0][:, 0] = 1
        uv[0][:, -1] = 1
        #neumann on horizontal sides, cell flux along the horizontal sides is none: u = u (this is especially important for when the wound vertically extends and "touches" the top and bottom of the grid)
        uv[0][0, :] = uv[0][1, :]
        uv[0][-1, :] = uv[0][-2, :]

        if i % 50 == 0: #appending every 50 iterations
            uarr_updates.append(np.copy(uv[0]))
            varr_updates.append(np.copy(uv[1]))
    
    return uarr_updates, varr_updates

def animate_plot(single_integrated_array, N):
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
    	fig, update, interval=150, blit=True, frames = len(single_integrated_array), repeat = False
	)
    plt.title(f"2D Fisher-KPP Model", fontsize=19)
    plt.show()



if __name__ == "__main__":

    uv = create_array(N, shape = "oval")
    #plt.imshow(uv[0], cmap='gray', origin='lower')
    #plt.show()

    ut, vt = fisher_kpp(uv)

    uarr_updates, varr_updates = numerical_integration_explicit_eulers(uv)

    animate_plot(uarr_updates, N)
    