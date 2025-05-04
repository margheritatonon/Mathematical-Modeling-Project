import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

D = 1
r = 1
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

if __name__ == "__main__":
    uv = create_array(N, shape = "oval")
    print(uv[0])
    plt.imshow(uv[0], cmap='gray', origin='lower')
    plt.show()
    