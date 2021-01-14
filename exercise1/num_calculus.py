""" comments for the file here """
# import needed packages, e.g., import numpy as np
import numpy as np


def first_derivative(function, x, dx):
    """
    Function for evaluating the first derivative of
    an input function f at point x. Error term O(h^2).
    """
    return ((function(x+dx)-function(x-dx))/(2*dx))


def second_derivative(function, x, dx):
    """
    Function for evaluating the second derivative of
    an input function f at point x. Error term O(h^2).
    """
    return ((function(x+dx)+function(x-dx)-2*function(x))/dx**2)


def simpson_int(f, x):
    """
    Calculates the integral with Simpson method where the grid
    has even spaces. Error term is O(h^3)
    """
    N = len(x)-1
    h = x[1]-x[0]
    s = 0.
    for i in range(1, N, 2):
        s += (f[i-1] + 4*f[i] + f[i+1])
    s = s*(h/3)

    # Add a term if odd number of intervals
    if (N+1) % 2 == 0:
        s = (s + (h/12)*(-f[N-2]+8*f[N-1]+5*f[N]))

    return s


def monte_carlo_integration(fun, xmin, xmax, blocks=10, iters=100):
    """
    Calculates the integral using the Monte Carlo method.
    """
    block_values = np.zeros((blocks,))
    L = xmax-xmin

    # Calculate the function value at random point x and sum them together
    for block in range(blocks):

        for i in range(iters):
            x = xmin+np.random.rand()*L
            block_values[block] += fun(x)
        # Mean value of the function values
        block_values[block] /= iters
    # Sum all the blocks together and divide by amount of blocks
    I = L*np.mean(block_values)

    # Error
    dI = L*np.std(block_values)/np.sqrt(blocks)
    return I, dI


def test_first_derivative(x, dx, fd):
    """
    Function tests the correctness of the first
    derivative evaluation function being inside
    the error term O(h^2). Returns boolean.
    """
    return (abs(12*x-fd) < dx**2)


def test_second_derivative(dx, sd):
    """
    Function tests the correctness of the second
    derivative evaluation function being inside
    the error term O(h^2). Returns boolean.
    """
    return (abs(12-sd) < dx**2)


def test_simpson_int(I, h):
    """
    Function tests the preciseness of the simpson integral
    evaluation function being inside the error term
    O(h^3). Returns boolean.
    """
    return (abs(1.-I) < h**2)


def test_monte_carlo_integration(I, dI):
    """
    Function tests the preciseness of the monte carlo integral
    evaluation function being inside the error term 2*dI for 95% of the time.
    """
    return (abs(1.-I) < dI)


def main():

    # function used for testing
    def function(x): return 6*x**2-5

    # precision
    dx = 0.001

    fd = first_derivative(function, 3, dx)
    sd = second_derivative(function, 4, dx)

    print(fd)
    print(test_first_derivative(3, dx, fd))
    print("\n")
    print(sd)
    print(test_second_derivative(dx, sd))

    # Simpson integral
    x = np.linspace(0, np.pi/2, 100)
    f = np.sin(x)
    I = simpson_int(f, x)
    h = x[2]-x[1]
    print("\n")
    print(I)
    print(test_simpson_int(I, h))

    # Monte Carlo integral
    def func(x): return np.sin(x)
    I, dI = monte_carlo_integration(func, 0., np.pi/2, 10, 100)
    print("\n")
    print("Integrated value: {0:0.5f} +/- {1:0.5f}".format(I, 2*dI))
    print(test_monte_carlo_integration(I, 2*dI))


if __name__ == "__main__":
    main()
