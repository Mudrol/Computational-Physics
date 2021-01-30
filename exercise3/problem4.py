"""
FYS-4096 Computational Physics: Exercise 3, Problem 4: Electric field

This program calculates the electric field due to a uniformly charged rod
of length L.

Made by: Matias Hiillos
"""

import numpy as np
from scipy.integrate import simps
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

L = 10. # Rod length
Q = 1e-9 # Positive charge
lmb = Q/L # Line charge density
eps_0 = 8.86418782e-12 # Vaccuum permittivity

def field_diff(r,eps_0,lmb):
    """
    Returns function for electric field

    r: distance between two points
    eps_0: vaccuum permittivity
    lmb: lambda, line charge density
    """
    return (1/(4*np.pi*eps_0)*lmb/r**2)


def electric_field_at_point(rod,x,rp):
    """
    Calculates the electric field by the charged rod at a point in 2d-space

    rod: rod coordinates
    x: x-coordinates for integrating
    rp: point to calculate the electric field at
    """
    # Calculate distance between the point and the rod points
    r = np.transpose(cdist(rp,rod))

    # Calculate the direction of the points (x-axis)
    rdirx = np.ones(len(rod[:,0]))
    for i in range(len(rdirx)):
        rodx = rod[i,0]
        if rp[0,0] - rodx < 0:
            rdirx[i] = -1

    # Calculate the electric field by the rod at the point with Simpson integral
    field_x = simps(rdirx*field_diff(r[:,0],eps_0,lmb),x[:,0])
    field_y = simps(field_diff(r[:,0],eps_0,lmb),x[:,0])*rp[0,1]

    return field_x,field_y

def electric_field_xy_plane(rod,x,y):
    """
    Calculates the electric field in the xy-plane due to the rod

    rod: rod coordinates
    x: x-coordinates of the plane
    y: y-coordinates of the plane
    """
    fieldx = np.zeros(shape=(len(x),len(y)))
    fieldy = np.zeros(shape=(len(x),len(y)))
    
    # Loop through every gridpoint
    for i in range(len(x)):
        for j in range(len(y)):

                # Compute the point
                rp = np.array([x[i][j],y[i][j]]).reshape(1,-1)

                # Calculate field x and y component at each point
                fieldx[i][j],fieldy[i][j] = electric_field_at_point(rod,x[i].reshape(-1,1),rp)
    return fieldx, fieldy



def test_field_diff_point_accurate(d):
    """
    Analytical answer to the electric field at point of the test case
    """
    return (lmb/(4*np.pi*eps_0)*(1/d-1/(d+L)))

def main():
    """
    Creates 'experimental data' and estimates values of the function
    using CHS -interpolation.
    """

    d = 6 # Used for point
    rp = np.array([L/2+d,0]).reshape(1,-1) # Point in space
    # Create the rod points
    x = np.linspace(-L/2,L/2,20).reshape(-1,1)
    y = np.zeros(len(x)).reshape(-1,1)
    # Add the points together
    rod = np.concatenate((x,y),axis=1)

    # Calulate electric field at point
    fieldx,fieldy = electric_field_at_point(rod,x,rp)

    # Check the result to be inside h^3
    E_acc = test_field_diff_point_accurate(d)
    if np.abs(fieldx-E_acc) < np.diff(x[:,0])[0]**3 and \
       np.abs(fieldy - 0) < np.diff(x[:,0])[0]**3:

        print("Field at point estimate is accurate!")
    else:
        print("Field at point estimate is NOT accurate!!")

    # Calculate electric field at xy-plane
    xx = np.linspace(-10,10,20).reshape(-1,1)
    yy = np.linspace(-10,10,20).reshape(-1,1)
    X, Y = np.meshgrid(xx,yy)
    fieldx,fieldy = electric_field_xy_plane(rod,X,Y)

    # Plotting
    plt.quiver(X,Y,fieldx,fieldy)
    plt.plot(x[:,0],y[:,0],'k-',linewidth=5)
    plt.title("Electric field due to a uniformly charged rod")
    plt.show()

if __name__ == "__main__":
    main()