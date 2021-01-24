"""
FYS-4096 Computational Physics: Exercise 2, Problem 1: Integration

Made by: Matias Hiillos
"""

import numpy as np
from num_calculus import simpson_integration
from scipy.integrate import simps


def integral_c(ra, rb):

    def psi(x,y,z): return (np.exp(-(np.sqrt(x**2+y**2+z**2)))/(np.sqrt(np.pi)))
    def r(x,y,z): return (np.sqrt(x**2+y**2+z**2))
    x,y,z = np.linspace(-10,10,100)

    # Function to be integrated
    fc = (psi(x-ra[0],y-ra[1],z-ra[2]))**2/(r(x-rb[0],y-rb[1],z-rb[2]))
    

def main():
    
    # Integrating functions in a)

    # Instead of inf integrate to 10, after that point the value of the integral
    # is close to 0.
    x1 = np.linspace(0,10,100)

    # 1/x is problematic for 0, so calculate 0 at 5-decimal precision
    x2 = np.linspace(0.00001,1,10)

    x3 = np.linspace(0,5,500)

    def f1(x): return x**2*np.exp(-2*x)
    def f2(x): return (np.sin(x)/x) 
    def f3(x): return (np.exp(np.sin(x**3)))

    int1 = simpson_integration(f1,x1)
    int2 = simpson_integration(f2,x2)
    int3 = simpson_integration(f3,x3)

    print("Part a) integrals:")
    print("{:.4f}".format(int1))
    print("{:.4f}".format(int2))
    print("{:.4f}".format(int3))


    # Integral of b), using the simps function from scipy.integrate

    # Needs to have same amount of elements (for every point there is x and y).
    x4 = np.linspace(0,2,100)
    y4 = np.linspace(-2,2,100)

    def f4(x,y): return (x*np.exp(-np.sqrt(x**2+y**2)))

    # Reshape the function as a NxN matrix
    fun4 = f4(x4.reshape(-1,1),y4.reshape(1,-1))

    # Integrate twice, first over x for every value to get the integrated
    # value for every y value, and then integrate over y
    int4 = simps([simps(fun4_x,x4) for fun4_x in fun4],y4)

    print("")
    print("Part b) integral:")
    print("{:.4f}".format(int4))


    # Integral of c)
    # Goal: calculate the value of the integral 

    # 5 test points
    testpts = [[0,0,0], [1,0,0], [1,1,1], [1,2,0]]





if __name__ == "__main__":
    main()