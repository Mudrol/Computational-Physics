"""
FYS-4096 Computational Physics: Exercise 2, Problem 4 b)
Steepest descent

Made by: Matias Hiillos
"""
import numpy as np
from num_calculus import numerical_gradient

def steepest_descent(f,x_init,a=0.9,tol=1e-4,counter_max=1000):
    """
    Steepest descent (not optimized)
    f: function taking x as an argument
    x_init: initial guess
    a: multiplier for gradient
    tol: tolerance for the search (default 1e-4)
    counter_max: maximum number of iterations
    """

    # Counter for max iterations
    counter = 0
    x = 1.0*x_init
    ngrad = 2*tol

    # Loop until we find maxima/minima inside tolerance or too many iterations
    while ngrad>tol and counter<counter_max:

        # Calculate the gradient at point x
        grad = numerical_gradient(f,x,tol)

        # Calculate the total gradient
        ngrad = np.sqrt(sum(grad**2))

        # Calculate the step to the guess
        dx = -a*grad
        x += dx
        counter += 1
    return x,counter

def test_steepest_descent():
    """
    Test routine for N-dimensional steepest descent.
    """
    N = 10
    x = np.linspace(-10,10,100)
    x_init = np.ones(N)
    dx = 1e-4
    xmin_est, count = steepest_descent(test_steepest_descent_fun,x_init)
    xmin_exact = test_steepest_descent_fun_min()
    err = np.abs(xmin_est-xmin_exact)
    i = 0
    for dim in err:
        if(dim>dx):
            print("Steepest descent evaluation is NOT ok!")
            return False
        i+=1
    print("Steepest descent evaluation is OK")
    return True


def test_steepest_descent_fun(x):
    """
    Test function for steepest descent
    """
    return x**2

def test_steepest_descent_fun_min():
    """
    This is the analytical answer for the test function for steepest descent
    """
    return np.zeros(10)

def main():
    test_steepest_descent()

if __name__ == "__main__":
    main()