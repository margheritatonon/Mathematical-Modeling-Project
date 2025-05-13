#this is what used to be main with no periodic boundary conditions (explosions at the edges)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#defining the parameters
lambd = 6 #g parameter
alpha = 1 #parameter for diffusion
rho = 11 #the radius for nonlocal integration
L = 200 #size of domain --> but halved (because domain goes from -L to L)
dx = 0.5 #spatial integration/derivative step
dt = 0.01 #time step

#VARIABLE DESCRIPTION:
#in the equations:
    # n represents the cell density, t is the time, and x is the spatial position. We are in 1 dimension so x is just one dimensional (not a vector).

#in the code:
    #narr is the array of cell densities. the values inside must range from 0 to 1.
        #the length of the narr array depends on dx. 
            #if dx = 1, the length is 2L. if dx = 0.5, the length is 2*2L. if dx = 0.25, the length is 4*2L, etc.
    #integrated_values is the result of the integral_beforen function. also a 2L/dx length array.


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

def integral_beforen(narr, dx=dx, rho = rho):
    """
    Computes the actual integral of the equation.
    Should return a 2L/dx length 1D array.
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
    integrated_values = np.zeros_like(narr) #this should be a 2L/dx length array.

    g_function_full = narr * (lambd - narr) #this is the entire g function, which we later need to slice

    for i in range(len(narr)):
        #LEFT part of the integral
        start = max(i - radius, 0)
        end = min(i + radius + 1, len(narr))
        g_neighbors = g_function_full[start:end] #this is one part of the integrand

        #putting together the integral:
        kernel_start = radius - (i - start) #we need to compute which part of the kernel to use so that it matches length of g_function
        kernel_end = kernel_start + (end - start)
        h_slice = h_kernel[kernel_start:kernel_end]

        integrand = g_neighbors * h_slice 
        integrated_values[i] = np.trapz(integrand, dx=dx)
    
    return integrated_values

#now, we multiply by n.
def by_n(narr, integrated_values):
    """
    Defines the expression that is used before the partial derivative wrt x is taken.
    """
    return narr * integrated_values

#now, we need to compute the partial derivative of this expression with respect to x.
def partial_wrt_x(expr, dx=dx):
    """
    Computes the partial derivative with respect to x.
    expr is the output of by_n. it is a 2L/dx length array, representing the expression that we need to take a partial derivative of.
    This function should also return a 2L/dx array.
    """
    #we use a finite difference approximation for the first derivative.
    left = np.roll(expr, 1)
    right = np.roll(expr, -1)
    return (right - left) / (2 * dx) #is of length 2L/dx
        
#now, we define a function that combines all of these three components together.
def pde(diffusion_term, advection_term, reaction_term, alpha=alpha):
    """
    Defines the PDE.
    Parameters:
        diffusion_term is a 2L/dx array that is computed with the laplacian function.
        advection_term is the result of the by_n function (integral expression multiplied by n). also 2L/dx
        reaction_term is the result of the f function. also 2L/dx
        alpha is the diffusion coefficient.
    Returns a 2L/dx array.
    """
    expression = alpha * diffusion_term -  advection_term + reaction_term
    return expression


#now we simulate the process:
def simulation(initial_n_condition, T, dx=dx, dt=dt):
    """
    Uses the explicit euler's scheme to simulate the PDE evolution.
    Ideally, it returns a list of n at different time points t (will then be used for plotting and animations).
    Parameters:
        initial_n_condition : the initial condition array, length 2L/dx.
        T is the time we wish to integrate over.
        xvals is
        dt: the time step
        dx: the spatial step
    Returns an array (2D) of cell density n at different values of T.
    """
    steps = int(T/dt)
    length = len(initial_n_condition)
    sol = np.zeros((steps, length)) #initializing where step solutions will be stored
    sol[0] = initial_n_condition.copy() #ensuring the first element is the initial condition

    #EULERS METHOD:
    for t in range(1, steps):
        current_narr = sol[t-1] #this is the array of cell densities of the (previous) step

        #we need to update the diffusion, reaction, and advection terms at every time step.
        diff_term = laplacians(current_narr)
        reac_term = f(current_narr)

        #advection term is more complicated:
        integratedvals = integral_beforen(current_narr)
        expr = by_n(current_narr, integratedvals)
        adv_term = partial_wrt_x(expr)

        #update step
        sol[t] = current_narr + dt * pde(diff_term, adv_term, reac_term)

    return sol #this is now a 2D array, where we have "steps" rows and columns of size 2L/dx

def animate_solution(sol, interval=100):
    fig, ax = plt.subplots()
    line, = ax.plot(sol[0])
    ax.set_ylim((-1,2))

    def update(frame):
        line.set_ydata(sol[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(sol), 10),
                                  interval=interval, blit=True)
    
    plt.show()

def plot_snapshots(sol, dt, times, L=L, dx=dx):
    """
    Plots the solution at given times.
    """
    x_vals = np.arange(-L, L+ dx, dx)
    indices = [min(int(t / dt), sol.shape[0] - 1) for t in times]
    fig, axs = plt.subplots(1, len(times), figsize=(4 * len(times), 4))

    for ax, idx, t in zip(axs, indices, times):
        ax.plot(x_vals, sol[idx])
        ax.set_title(f"t = {t}")
        ax.set_ylim(min(sol[idx])-0.1, max(sol[idx])+0.1)
        ax.set_xlim(x_vals[0], x_vals[-1])
        ax.grid(True)

    plt.suptitle(f"lambda = {lambd}, rho = {rho}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    initial_n_condition = wound(L=L, dx=dx)
    simulation_arr = simulation(initial_n_condition, T=100, dx=dx, dt=dt)
    #print(simulation_arr) 
    #print(simulation_arr.shape)
    animate_solution(simulation_arr)
    plot_snapshots(simulation_arr, dt=dt, times=(0,10,20, 50, 70, 100))