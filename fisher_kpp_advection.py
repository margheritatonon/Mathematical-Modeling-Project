import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import integrate
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded



#defining parameters:
lambd = 7
alpha = 1 
rho = 8 #the radius for nonlocal integration
L = 200 #size of domain --> but halved (domain goes from -L to L)
dx = 0.25

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

def compute_integral_convolution(narr, rho, dx = dx):
    """
    Computes the nonlocal integral using convolution
    """
    r = int(rho / dx)
    xhat_vals = np.arange(-r, r + 1)
    h_vals = h(xhat_vals)
    h_vals *= dx

    #trapezoidal rule weights
    h_vals[0] *= 0.5
    h_vals[-1] *= 0.5

    #g(n) * h(xhat) convolved over space
    return convolve(g(narr), h_vals, mode='same')

def compute_nonlocal_integral_noconvolution(narr, xhat_vals, h_vals):
    """
    Computes the nonlocal integral without convolution.
    """
    g_n = g(narr)
    r = len(xhat_vals) // 2

    # Pad g(n) with zeros to handle boundary
    padded = np.pad(g_n, pad_width=r, mode='constant', constant_values=0)

    # Create an output array
    integrated = np.zeros_like(narr)

    for i in range(len(narr)):
        window = padded[i:i + 2 * r + 1]
        integrated[i] = np.sum(window * h_vals)

    return integrated

def compute_nonlocal_integral_fast(narr, rho, x_vals, dx=dx):
    xhat_vals = np.arange(-rho, rho + dx, dx)
    h_vals = h(-xhat_vals)  # flip direction to match convolution kernel
    
    # Apply trapezoidal weights (dx is built-in to integrate.trapezoid below)
    h_vals[0] *= 0.5
    h_vals[-1] *= 0.5

    # Use g(n)
    g_n = g(narr)

    # Extend domain slightly to avoid extrapolation errors
    interp_gn = interp1d(x_vals, g_n, bounds_error=False, fill_value=0.0)

    # Build full x + x̂ grid
    x_matrix = x_vals[:, None] + xhat_vals[None, :]
    g_matrix = interp_gn(x_matrix)

    # Multiply by h and integrate
    integrand = g_matrix * h_vals[None, :]
    result = integrate.trapezoid(integrand, dx=dx, axis=1)

    return result



def rhs(narr, rho, xvals, dx = dx):
    """
    Defines the right hand side of the PDE 
    """
    #integrated = compute_integral_convolution(narr, rho, dx=dx)
    integrated = compute_nonlocal_integral_fast(narr, rho, xvals, dx=dx)
    
    #advection term
    adv_term = partial_wrt_x(before_singlepartial(integrated, narr), dx=dx)

    #diffusion term
    diff_term = alpha * laplacians(narr, dx=dx)

    #growth term
    growth_term = f(narr)

    return diff_term - adv_term + growth_term

def simulate(initial_n_cond, T, xvals, dt = 0.1, rho=rho):
    """
    Simulates the PDE for time T, using Explicit Euler's Method.
    """
    steps = int(T / dt)
    nx = len(initial_n_cond)
    sol = np.zeros((steps, nx))
    sol[0] = initial_n_cond.copy()

    for t in range(1, steps):
        sol[t] = sol[t-1] + dt * rhs(sol[t-1], rho, xvals) #eulers method
        if np.any(np.isnan(sol[t])) or np.any(np.isinf(sol[t])):
            print(f"Numerical instability at time step {t}, time {t*dt}")
            break
    
    return sol

def simulate_imex(initial_n, T, xvals, dt, dx, rho):
    steps = int(T / dt)
    nx = len(initial_n)
    sol = np.zeros((steps, nx))
    sol[0] = initial_n.copy()

    # Precompute diffusion matrix (implicit part)
    alpha_dt_dx2 = alpha * dt / dx**2
    lower = np.full(nx-1, -alpha_dt_dx2)
    diag  = np.full(nx, 1 + 2 * alpha_dt_dx2)
    upper = np.full(nx-1, -alpha_dt_dx2)

    # Neumann BC
    diag[0] = diag[-1] = 1 + alpha_dt_dx2
    upper[0] = lower[-1] = -alpha_dt_dx2

    ab = np.zeros((3, nx))
    ab[0,1:] = upper
    ab[1,:] = diag
    ab[2,:-1] = lower

    for t in range(1, steps):
        n_prev = sol[t-1]
        # compute nonlocal advection and growth terms
        integral = compute_nonlocal_integral_fast(n_prev, rho, xvals, dx=dx)
        adv = partial_wrt_x(n_prev * integral, dx=dx)
        growth = f(n_prev)

        rhs_vec = n_prev + dt * ( -adv + growth )

        sol[t] = solve_banded((1, 1), ab, rhs_vec)

        if np.any(np.isnan(sol[t])) or np.any(np.isinf(sol[t])):
            print(f"Numerical instability at t = {t*dt}")
            break

    return sol


def animate_solution(sol, interval=100):
    fig, ax = plt.subplots()
    line, = ax.plot(sol[0])

    def update(frame):
        line.set_ydata(sol[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(sol), 10),
                                  interval=interval, blit=True)
    
    plt.show()


def plot_snapshots(sol, dt, times, x_vals):
    """
    Plots the solution at given times.
    """
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
    dt = 0.01
    x_vals = np.arange(-L, L + dx, dx)
    myx0 = wound(L=L)
    np.random.seed(42)
    myx0 += 1e-3 * np.random.randn(*myx0.shape)
    plot_initialcond = False
    if plot_initialcond == True:
        plt.plot(np.arange(-200, 201), myx0)
        plt.show()

    sol1 = simulate(myx0, T=300, xvals = np.arange(-L, L+ dx, dx), dt=dt, rho=rho)
    sol = simulate_imex(myx0, T=300, xvals=x_vals, dt=0.1, dx=dx, rho=rho)

    animate_solution(sol)

    plot_times = [0, 1, 2, 10, 70]  #avoid edge case
    plot_snapshots(sol, dt=dt, times=plot_times, x_vals=x_vals)


    deb = False
    if deb:
        i_conv = compute_integral_convolution(myx0, rho, dx=dx)
        i_fast = compute_nonlocal_integral_fast(myx0, rho, x_vals, dx=dx)

        plt.plot(x_vals, i_conv, label='Convolution')
        plt.plot(x_vals, i_fast, label='Fast')
        plt.legend()
        plt.title("Comparison of nonlocal integral methods at t=0")
        plt.show()