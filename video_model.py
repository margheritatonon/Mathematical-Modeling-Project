import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#MODEL TYPES:
#simple model: ut = D * lap_u + gamma * u^2 * lap_u
#cell diffusion (only): ut = D*lap_u
#diffusion and growth: ut = D*lap_u + p*u*(1-u)
#chemotaxis (only): ut = -X*lap_u*c*u --> need to figure out what c is here 

#defining parameter values
D = 0.1
gamma = 1
p = 0.05
region_length = 40
model = "diffusion and growth" #select the type of model you want to look at: "simple", "diffusion", "diffusion and growth"

def create_array(N:int, region_length = region_length): 
    """
    Returns an initial condition 1D array with noise of length N.
    We can simulate different initial cell patterns here.
    Parameters:
        N: the number of spatial points in the discretization
    """
    u = np.ones((1, N))

    #random noise:
    #u = u + 0.5*np.random.uniform(0, 1, (1, N)) #not sure if we still need to add this noise here.

    #central bump:
    #u[0, N//2 - 5:N//2 + 5] += 2 

    #gaussian bump: the bump flattens and expands, simulating diffusive spreading with slight focusing
    #x = np.linspace(-2, 2, N)
    #u = np.exp(-x**2)[None, :] + 0.1

    #a strong bump:
    #u[0, N//3:N//3+5] += 1.5

    #cells around edges of region (not sure this is the right approach, because im not sure if this code actually simulates the movement of the cells):
    u = np.ones((1, N)) 
    u[0, :4] = 2.0 + 0.2 * np.random.rand(4) #4 is the edge width: this can be changed depending on how big you want the edges of the wound to be
    u[0, -4:] = 2.0 + 0.2 * np.random.rand(4)
    u += 0.01 * np.random.rand(1, N) #noise

    return u

def spatial_part(u, model=model, dx = 1, gamma = gamma, p = p, D = D):
    """
    Implements a 1D finite difference numerical approximation to integrate the spatial part of the reaction-diffusion equations.
    Parameters:
        u: a 1D array of initial conditions for u.
        dx: the spatial step (defaults to 1).
    Returns:
        the PDE u_t
    """
    #computing laplacians - we are applying the 1D finite difference numerical scheme
    u_plus = np.roll(u, shift = dx)
    u_min = np.roll(u, shift = -dx)
    lap_u = u_plus - 2*u + u_min

    #using this in the equation (the simple one)
    if model == "simple":
        ut = D * lap_u + gamma * (u**2) * lap_u
    elif model == "diffusion":
        ut = D * lap_u
    elif model == "diffusion and growth":
        ut = D * lap_u + p * u * (1 - u)

    return ut

def numerical_integration(u:np.array, ut:np.array, dt:float=0.01):
    """
    Numerically integrates array ut obtained from spatial_part function using Explicit Euler's method.
    Parameters:
        dt: float specifying the time step for numerical integration.
    Returns a list with 100 elements (frames).
    """
    uarr_updates = []

    for i in range(50000): 
        ut = spatial_part(u)
        #updating with explicit eulers method
        if i % 500 == 0: #appending every 500 iterations
            uarr_updates.append(np.copy(u[0]))
        u = u + ut * dt

        #boundary conditions:
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
    
    return uarr_updates

def plot_static(region_length, uarr_updates):
    """
    Creates a static plot of the last frame of animation of x versus v. 
    """
    #static plot:
    x_arr = np.linspace(0, region_length, region_length)
    fig, ax = plt.subplots(1, 1)
    plt.plot(x_arr, uarr_updates[-1])
    ax.set_xlim((0, region_length))
    ax.set_ylim((np.min(uarr_updates) * 0.75, np.max(uarr_updates) * 1.5)) #so we can see the plot easier
    plt.xlabel("x", fontsize = 15)
    plt.ylabel("u(x)", fontsize = 15)
    plt.title(f"Final Animation Frame", fontsize = 20)
    plt.show()
    plt.close()

def animate_plot(region_length, uarr_updates):
    """
    Animates the plot of the numerically integrated solution.
    """
    fig, ax = plt.subplots(1, 1)
    x_arr = np.linspace(0, region_length, region_length) 
    (plot_v,) = ax.plot(x_arr, uarr_updates[0]) 

    def update(frame):
        plot_v.set_ydata(uarr_updates[frame])  
        return plot_v,

    ani = animation.FuncAnimation(fig, update, frames=len(uarr_updates), interval=200, blit=True)
    plt.xlabel("x")
    plt.ylabel("v")
    ax.set_xlim((0, region_length))
    ax.set_ylim((0, 4))
    plt.show()
    plt.close()

def plot_initial_final(region_length, uarr_updates):
    """
    Creates a plot comparing the initial and final frame of the system.
    """
    x_arr = np.linspace(0, region_length, region_length)
    fig, ax = plt.subplots(1, 1)
    plt.plot(x_arr, uarr_updates[0], label="Initial State", color="blue")
    plt.plot(x_arr, uarr_updates[-1], label="Final State", color="red")
    
    ax.set_xlim((0, region_length))
    ax.set_ylim((np.min(uarr_updates) * 0.75, np.max(uarr_updates) * 1.5))
    plt.xlabel("x", fontsize=15)
    plt.ylabel("u(x)", fontsize=15)
    plt.title(f"Initial Versus Final State", fontsize=20)
    plt.legend()
    plt.show()
    plt.close()

#TODO: add a function that plots different frames of the animation in a static way (e.g. 2 by 2 plots of 4 frames over time)

if __name__ == "__main__":
    u = create_array(region_length) #creating u array (initial condition)
    ut = spatial_part(u) #obtaining the differential equation left hand side
    uarr_updates = numerical_integration(u, ut)
    print(uarr_updates)
    plot_static(region_length, uarr_updates)
    animate_plot(region_length, uarr_updates)
    plot_initial_final(region_length, uarr_updates)



    
