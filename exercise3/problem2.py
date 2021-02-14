"""
FYS-4096 Computational Physics: Exercise 3, Problem 2: 2D interpolation 
revisited.

Made by: Matias Hiillos
"""

import numpy as np
import spline_class
import matplotlib.pyplot as plt


def function(x,y):
    """
    Function for 2d interpolation testing
    """
    return (x+y)*np.exp(-np.sqrt(x**2+y**2))


def main():
    """
    Creates 'experimental data' and estimates values of the function
    using CHS -interpolation.
    """
    pts = 30
    x = np.linspace(-2,2,pts)
    y = np.linspace(-2,2,pts)
    X,Y = np.meshgrid(x,y)

    # Function for generating 'experimental data'
    Z = (X+Y)*np.exp(-np.sqrt(X**2+Y**2))
    spl2d = spline_class.spline(x=x,y=y,f=Z,dims=2)

    # Upper limit of 1 to make the interpolation more accurate
    x = np.linspace(0,1,10)
    y = 2*x**2
    xx = np.linspace(0,1,100)
    yy = 2*xx**2

    # Evaluate the function, taking only the diagonal values
    f_eval = np.diag(spl2d.eval2d(x,y))

    # Exact function along the path
    f_exact = function(xx,yy)

    # Plotting the data
    plt.figure()
    plt.plot(x,f_eval,'r--', label="Interpolated function")
    plt.plot(xx,f_exact,'b', label="Exact function")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()