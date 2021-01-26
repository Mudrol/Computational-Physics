"""
FYS-4096 Computational Physics: Exercise 3, Problem 1: 2D Integral revisited
Calculating the integral using 20 different grid spacings. It appears that
having an odd-value as points gives more accurate result.

Made by: Matias Hiillos
"""

import numpy as np
from scipy.integrate import simps


def integral_2d(f,x,y):
    """
    Function for calculating the integral
    """

    # Reshape the function for broadcasting
    func = f(x.reshape(-1,1),y.reshape(1,-1))

    # Integrate first over x, then over y
    res = simps([simps(func_x,x) for func_x in func],y)
    return res


def function(x,y):
    """
    Function used in integration
    """
    return (x+y)**2*np.exp(-np.sqrt(x**2+y**2))


def main():

    # Testing with different grids
    for pts in range(1,20,1):
        x = np.linspace(0,2,pts)
        y = np.linspace(-2,2,pts)
        res = integral_2d(function,x,y)
        print("{:.4f}".format(res), "with", pts, "points")


if __name__ == "__main__":
    main()