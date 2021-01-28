"""
FYS-4096 Computational Physics: Exercise 3, Problem 4: Electric field 

Made by: Matias Hiillos
"""

import numpy as np
from scipy.integrate import simps
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

L = 30. # Rod length
Q = 1.602e-19 # Positive charge
lmb = Q/L # Line charge density
eps_0 = 8.86418782 # Vaccuum permittivity

def field_diff(r,eps_0,lmb):
    """
    Returns function for electric field
    """
    return (1/(4*np.pi*eps_0)*lmb/r**2)


def test_field_diff_point(rod,x):
    """
    Calculates the electric field by the charged rod at a point in space
    rod = rod points
    x = x-axis points
    """

    d = 6 # Used for test point
    rp = np.array([L/2+d,0]).reshape(1,-1) # Point in space

    # Calculate distance between the point and the rod points
    r = np.transpose(cdist(rp,rod))

    # Calculate the electric field by the rod at the point with Simpson integral
    E_approx = simps(field_diff(r[:,0],eps_0,lmb),x[:,0])

    # Check the result to be inside h^3
    E_acc = test_field_diff_point_accurate(d)
    working = False
    if np.abs(E_approx-E_acc) < np.diff(x[:,0])[0]**3:
        print("Field at point estimate is accurate!")
        working = True
    else:
        print("Field at point estimate is NOT accurate!!")
        working = False
    return working

def test_field_diff_point_accurate(d):
    """
    Analytical answer to the electric field at point of the test
    """
    return (lmb/(4*np.pi*eps_0)*(1/d-1/(d+L)))

def main():
    """
    Creates 'experimental data' and estimates values of the function
    using CHS -interpolation.
    """

    # Create the rod points
    x = np.linspace(-L/2,L/2,100).reshape(-1,1)
    y = np.zeros(len(x)).reshape(-1,1)
    # Add the points together
    rod = np.concatenate((x,y),axis=1)

    # Test function for point
    test_field_diff_point(rod,x)

 
if __name__ == "__main__":
    main()