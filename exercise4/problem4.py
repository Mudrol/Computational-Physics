"""
FYS-4096 Computational physics: Exercise 4, problem 4: Electron density along a line

This program calculates the electron density along two lines.
Includes transforming the coordinates into alpha space, and periodic boundary
conditions.

Made by: Matias Hiillos
"""

import numpy as np
import spline_class
import matplotlib.pyplot as plt
from read_xsf_example import read_example_xsf_density

def electron_density_line(rho, lattice, grid):
    """
    Calculates the electron density of two different lines by first
    changing the coordinates to alpha coordinates.
    """
    pts = 500

    # Create grid points for spline class in alpha space
    a1 = np.linspace(0,1,grid[0])
    a2 = np.linspace(0,1,grid[1])
    a3 = np.linspace(0,1,grid[2])


    # Create spline object
    spl = spline_class.spline(x=a1,y=a2,z=a3,f=rho,dims=3)


    # Compute points for the paths, in alpha spzce
    # Also add periodic boundary conditions by using modulo operator

    # Points in alpha space are represented as r = A*alpha, where
    # alpha_i in each direction is between 0 and 1.
    r1_0 = np.array([-1.4466,1.3073,3.2115])
    r1_1 = np.array([1.4361,3.1883,1.3542])

    r2_0 = np.array([2.9996,2.1733,2.1462])
    r2_1 = np.array([8.7516,2.1733,2.1462])

    # Calculate the points in alpha space, A*alpha = r
    r10_a = np.linalg.solve(lattice,r1_0)
    r11_a = np.linalg.solve(lattice,r1_1)

    r20_a = np.linalg.solve(lattice,r2_0)
    r21_a = np.linalg.solve(lattice,r2_1)

    t = np.linspace(0,1,pts)
    rho_line1 = np.zeros(pts)
    rho_line2 = np.zeros(pts)
    i = 0

    # Calculate the electron density at each point of the lines in alpha space
    for point in t:
        r1_a = (r11_a-r10_a)*point + r10_a
        r2_a = (r21_a-r20_a)*point + r20_a

        # Map the r in alpha space inside the primitive cell
        r1_a = np.mod(r1_a,1)
        r2_a = np.mod(r2_a,1)

        rho_line1[i] = spl.eval3d(r1_a[0],r1_a[1],r1_a[2])
        rho_line2[i] = spl.eval3d(r2_a[0],r2_a[1],r2_a[2])
        i+=1

    # Plotting

    fig1 = plt.figure()
    plt.plot(t,rho_line1)
    plt.title("Electron density along the first line")
    plt.xlabel("r length")
    plt.ylabel("Electron density")

    fig2 = plt.figure()
    plt.plot(t,rho_line2)
    plt.title("Electron density along the second line")
    plt.xlabel("r length")
    plt.ylabel("Electron density")

    plt.show()
    


    
def main():

    # Read the file
    filename = 'dft_chargedensity2.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)

    electron_density_line(rho,lattice,grid)

    
if __name__=="__main__":
    main()



