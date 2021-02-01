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

    # 500 points takes too long to compute
    pts = 50

    # Compute points for the path
    r_0 = np.array([0.1,0.1,2.8528])
    r_1 = np.array([4.45,4.45,2.8528])
    x = np.linspace(r_0[0],r_1[0],pts)
    y = np.linspace(r_0[1],r_1[2],pts)
    z = np.linspace(r_0[2],r_1[2],pts)

    # Create grid points for spline class
    xx = np.linspace(0.,lattice[0,0],grid[0])
    yy = np.linspace(0.,lattice[1,1],grid[1])
    zz = np.linspace(0.,lattice[2,2],grid[2])

    # Create spline object
    spl = spline_class.spline(x=xx,y=yy,z=zz,f=rho,dims=3)

    # Calculate the electron density values along path
    rho_line = spl.eval3d(x,y,z)


    # Plotting
    fig = plt.figure()
    plt.plot(x,np.diag(rho_line[:,:,0]))
    plt.title("Electron density along the line")
    plt.xlabel("x, y")
    plt.ylabel("Electron density")

    plt.show()
    


    
def main():

    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)

    electron_density_line(rho,lattice,grid)

    
if __name__=="__main__":
    main()



