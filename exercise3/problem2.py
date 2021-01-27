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
    x = np.linspace(0,2,100)
    y = 2*x**2

    # Evaluate the function, taking only the diagonal values
    f_eval = np.diag(spl2d.eval2d(x,y))

    # Exact function along the path
    f_exact = function(x,y)

    # Plotting the data
    plt.figure()
    plt.plot(x,f_eval,'o', label="Interpolated function")
    plt.plot(x,f_exact,'r--', label="Exact function")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()