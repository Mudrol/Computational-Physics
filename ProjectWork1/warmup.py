"""
    FYS-4096 Computational Physics: Project Work 1.

    Warmup exercise

    This program calculates the 2D integral over an area using scipy's simps
    function.

    Made by: Matias Hiillos
"""

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

def integrate_2d(a1,a2,N):
    """
    This function integrates the function
    """
    N = 50
    a1 = np.array([1.2,0.1])
    a2 = np.array([0.6,1.])
    x = np.linspace(0,1,N)

    # Get values of function inside grid
    X,Y = np.meshgrid(x,x)
    XX = X*a1[0] + Y*a2[0]
    YY = X*a1[1] + Y*a2[1]
    Z = fun(XX,YY)

    # Integrate twice
    int_x = simps(Z, XX)
    res = simps(int_x, x)
    return res, XX, YY, Z


def fun(x,y):
    return ((x+y)*np.exp(-0.5*np.sqrt(x**2+y**2)))

def main():
    N = 500
    a1 = np.array([1.2,0.1])
    a2 = np.array([0.6,1.])

    res, X, Y, Z = integrate_2d(a1,a2,N)
    print("Value of integral: %.4f" % res)

    # Plot the integrand
    plt.contourf(X,Y,Z)
    plt.title(r"Integrand inside the area $\Omega$")
    plt.xlabel("x")
    plt.ylabel("y")


    plt.show()

if __name__=="__main__":
    main()
