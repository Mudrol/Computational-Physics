"""
    FYS-4096 Computational Physics: Project Work 1.

    Warmup exercise

    This program calculates the 2D integral over an area using scipy's simps
    function.

    Made by: Matias Hiillos
"""
import numpy as np
from scipy.integrate import simps

def integrate_2d():
    """
    This function integrates the function
    """
    N = 50
    a1 = np.array([1.2, 0.1])    
    a2 = np.array([0.6, 1.])    

    # Calculate all the gridpoints for the space given by the vectors
    grid = np.linspace(0,1,N)
    X,Y = np.meshgrid(grid,grid)

    XX = X*a1[0] + Y*a2[0] 
    YY = X*a1[1] + Y*a2[1]

    # Integrate twice
    res = simps(simps(fun(XX,YY),XX),grid)
    print(res)


def fun(x,y):
    return ((x+y)*np.exp(-0.5*np.sqrt(x**2+y**2)))

def main():

    integrate_2d()

if __name__=="__main__":
    main()
