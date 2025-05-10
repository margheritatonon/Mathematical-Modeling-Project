import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import integrate
from scipy.signal import convolve
from scipy.interpolate import interp1d


#defining parameters:
lambd = 7
alpha = 1 
rho = 8 #the radius for nonlocal integration
L = 200 #size of domain --> but halved (domain goes from -L to L)
dx = 0.25
D=1 #diffusion coefficient

#clarifying notation:
#when we perform the integral, we are integrating with respect to xhat for one specific value of x.
#xhat are basically the x values that are in a certain radius away from x.
#xhat is a displacement vector representing the relative position between two interacting cells
#therefore for all x values we need to compute this integral (i think)
#the integral is a nonlocal integral that aggregates information about the cell density n in a neighborhood of x ([x−rho,x+rho])
#so when we have n(x+xhat, t) in the integral, we are basically just integrating over a neighboring region.



#h(x) does not depend on time so we can calculate those values once and use them throughtout.


#we begin by discretizing the domain and defining the initial wound
#in the paper, the domain is 1D.
def wound(L=L, dx = dx):
    """
    This creates the -L to L initial condition array of the wound.
    This actually represents a cross section of the wound, viewed laterally.
    """
    x = np.arange(-L, L+dx, dx)
    x0 = 1.0 - 0.75 * np.exp(-(0.1 * x) ** 6) #the initial condition they define in the paper
    return x0


#defining a function as the expression inside the integral:
def g(n):
    """
    Defines the function g for the array of n values.
    """
    return n * (lambd - n) 

def h(xhat):
    """
    Defines the h(x) function for the array of values that neighbor the
    Parameters:
        xhat: the indices of the neighboring values of a certain value x.
    """
    return (0.1*np.arctan(0.2*xhat))/(np.arctan(2))

def precompute_h_kernel(rho, dx=dx):
    """
    This precomputes xhat and h(xhat) since they do not depend on time.
    """
    r = int(rho / dx)
    xhat_vals = np.arange(-r, r + 1)
    
    h_vals = h(xhat_vals)
    h_vals[0] *= 0.5
    h_vals[-1] *= 0.5
    h_vals *= dx

    return xhat_vals, h_vals

def A(narr, currx, rho, dx = dx):
    """
    This is the expression inside of the integral.
    Parameters:
        narr: the array of current n values
        currx: the x index we look at
        rho: the radius we consider
    """
    r = int(rho / dx)
    xhat_vals = np.arange(-r, r + 1)
    indices = currx + xhat_vals

    valid_mask = (indices >= 0) & (indices < len(narr))
    valid_indices = indices[valid_mask]
    padded_gn = np.zeros_like(xhat_vals, dtype=float)
    padded_gn[valid_mask] = g(narr[valid_indices])

    return padded_gn * h(xhat_vals)

    #we first take the neighbors of x that are in a radius of rho away
    #we therefore apply a mask
    #the thing is that its one dimensional so we just take the neighbors of x + rho, x-rho
    #indices = np.arange(min(0, currx-rho), min(currx+rho, len(narr))) #prevents errors in indexing, in case currx-rho < 0 or currx+rho > the total array length
    #now we index n with these indices
    #ns = narr[indices]
    #a = g(ns) * h(indices) #this is the expression that is inside of the integral
    #return a  #we will need to integrate a numerically, from -rho to rho.

def integrating_expression(to_integrate, rho, dx = dx):
    """
    Integrates the to_integrate array using the trapezoidal rule.
    """
    r = int(rho / dx)
    xhat_vals = np.arange(-r, r + 1) * dx  #the integration bounds
    result = integrate.trapezoid(to_integrate, xhat_vals)
    return result

def f(n):
    """
    Defining the f(n) function in the expression.
    """
    return n*(1-n)

def before_singlepartial(integrated, n):
    """
    Defines the expression before the partial derivative with respect to x is taken.
    """
    return n * integrated

def partial_wrt_x(expression, dx = dx):
    """
    Computing the partial derivative of the expression n*the integral in the formula.
    We use central differences to approximate the derivative.
    """
    derivative = np.zeros_like(expression)
    derivative[1:-1] = (expression[2:] - expression[:-2]) / (2 * dx)
    derivative[0] = (expression[1] - expression[0]) / dx
    derivative[-1] = (expression[-1] - expression[-2]) / dx

    return derivative

def laplacians(narr, dx=dx):
    """
    Defines the laplacian (second partial derivative wrt x) of narr, the cell density array.
    This is a 1D Laplacian (we therefore use a three point stencil)
    """
    lap = np.zeros_like(narr)
    lap[1:-1] = (narr[2:] - 2 * narr[1:-1] + narr[:-2]) / dx**2
    #neumann boundary conditions (zero flux)
    lap[0] = lap[1]
    lap[-1] = lap[-2]
    return lap

def compute_nonlocal_term(narr, rho, dx=dx):
    """
    Computes the nonlocal integral term for each point in narr.
    Returns the array of integrated values for each spatial point.
    """
    integrated_vals = np.zeros_like(narr)
    for i in range(len(narr)):
        integrand = A(narr, i, rho, dx)
        integrated_vals[i] = integrating_expression(integrand, rho, dx)
    return integrated_vals

def compute_nonlocal_term_convolution(narr, rho, dx=dx):
    r = int(rho / dx)
    xhat_vals = np.arange(-r, r + 1)
    
    kernel = h(xhat_vals)
    kernel[0] *= 0.5
    kernel[-1] *= 0.5
    kernel *= dx

    gn = g(narr)
    # pad with zeros to match boundary behavior
    pad_width = len(kernel) // 2
    padded_gn = np.pad(gn, pad_width, mode='constant')
    conv = np.convolve(padded_gn, kernel, mode='valid')  # result has same length as gn
    return conv


def timestep(n, alpha, D, rho, dt, dx=dx):
    """
    Performs a single time step update of the PDE, with Explicit Eulers.
    """
    integrated_vals = compute_nonlocal_term_convolution(n, rho, dx)
    advection = partial_wrt_x(before_singlepartial(integrated_vals, n), dx)
    reaction = alpha * f(n)
    diffusion = D * laplacians(n, dx)
    return n + dt * (diffusion - advection + reaction)

def run_simulation(L, dx, dt, tmax, alpha, rho, D=D):
    """
    Simulates the main loop of the simulation
    """
    x_vals = np.arange(-L, L + dx, dx)
    n = wound(L, dx)
    snapshots = [n.copy()]
    times = [0]

    for t in np.arange(dt, tmax + dt, dt):
        n = timestep(n, alpha, D, rho, dt, dx)
        if int(t / dt) % 10 == 0:
            snapshots.append(n.copy())
            times.append(t)
    return x_vals, snapshots, times


def animate_simulation(x_vals, snapshots, times):
    """
    Animates the result of the simulation
    """
    fig, ax = plt.subplots()
    line, = ax.plot(x_vals, snapshots[0])
    ax.set_ylim(0, 1.5)

    def update(frame):
        line.set_ydata(snapshots[frame])
        ax.set_title(f"Time = {times[frame]:.2f}")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=50)
    plt.show()

def plot_static_snapshots(x_vals, snapshots, times, time_points):
    """
    Plots static snapshots of the solution at desired time points
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, target_time in enumerate(time_points):
        idx = np.argmin(np.abs(np.array(times) - target_time))
        axes[i].plot(x_vals, snapshots[idx], lw=2)
        axes[i].set_title(f"t = {times[idx]:.2f}")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("n(x)")
        axes[i].grid(True)
    
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    dt = 0.01
    x_vals = np.arange(-L, L + dx, dx)
    myx0 = wound(L=L)
    plot_initialcond = False
    if plot_initialcond == True:
        plt.plot(np.arange(-200, 201), myx0)
        plt.show()
    
    tmax = 20
    x_vals, snapshots, times = run_simulation(L, dx, dt, tmax, lambd, alpha, rho, D)
    animate_simulation(x_vals, snapshots, times)
    
    plot_static_snapshots(x_vals, snapshots, times, time_points=[0, 5, 10, 20])


