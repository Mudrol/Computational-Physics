"""
FYS-4096 Computational Physics: Exercise 3, Problem 2: 2D interpolation 
revisited.

Made by: Matias Hiillos

TODO: Get it to work better
"""

import numpy as np
import linear_interp
import matplotlib.pyplot as plt

def integral_2d(f,x,y):
    """
    Function for calculating the integral
    """
    pass

def function(x,y):
    """
    Function for generating 'experimental data'
    """
    return (x+y)*np.exp(-np.sqrt(x**2+y**2))


def main():

    pts = 30
    x = np.linspace(-2,2,pts)
    y = np.linspace(-2,2,pts)
    X,Y = np.meshgrid(x,y)

    # Function for generating 'experimental data'
    Z = (X+Y)*np.exp(-np.sqrt(X**2+Y**2))
    lin2d = linear_interp.linear_interp(x=x,y=y,f=Z,dims=2)
    x = np.linspace(0,2,100)
    y = 2*x**2
    fun = lin2d.eval2d(x,y)

    # Exact function along the path
    f_exact = (x+2*x**2)*np.exp(-np.sqrt(3*x**2))

    # Plotting the data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,fun[:,10],'o', label="Interpolated function")
    ax.plot(x,f_exact,'r--', label="Exact function")
    ax.set_title('function')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()