"""
FYS-4096 Computational physics: Exercise 4, problem 2

This program calculates the electron density along a line

Made by: Matias Hiillos
"""

import numpy as np
import spline_class
import matplotlib.pyplot as plt
from read_xsf_example import read_example_xsf_density

def electron_density_line(rho, lattice, grid):

    pts = 500

    # Create grid points for spline class
    xx = np.linspace(0.,lattice[0,0],grid[0])
    yy = np.linspace(0.,lattice[1,1],grid[1])
    zz = np.linspace(0.,lattice[2,2],grid[2])

    # Create spline object
    spl = spline_class.spline(x=xx,y=yy,z=zz,f=rho,dims=3)

    # Create the line path
    r_0 = np.array([0.1,0.1,2.8528])
    r_1 = np.array([4.45,4.45,2.8528])
    t = np.linspace(0,1,pts)

    rho_line = np.zeros(pts)
    i = 0
    for val in t:

        r = (r_1-r_0)*val+r_0
        # Calculate the electron density values along path
        rho_line[i] = spl.eval3d(r[0],r[1],r[2])
        i += 1

    # Plotting
    fig = plt.figure()
    plt.plot(t,rho_line)
    plt.title("Electron density along the line")
    plt.xlabel("r length")
    plt.ylabel("Electron density")

    plt.show()
    


    
def main():

    # Read the file 
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)

    electron_density_line(rho,lattice,grid)

    
if __name__=="__main__":
    main()



