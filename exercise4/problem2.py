"""
FYS-4096 Computational physics: Exercise 4, problem 2

This program calculates the number of electrons in the simulation.

Made by: Matias Hiillos
"""

import numpy as np
from scipy.integrate import simps
from read_xsf_example import read_example_xsf_density

def number_of_electrons(rho, lattice, grid):
    """
    Calculates the number of electrons in a cell.
    """
    # Calculate the volume around a single block in grid using scalar triple
    # product: a*(bxc)
    block_volume = np.dot(np.cross(lattice[0]/grid[0],lattice[1]/grid[1]),lattice[2]/grid[2])

    # Calculate the amount of electrons at each block and sum them together
    N = np.sum(block_volume * rho)

    return N


def reciprocal_lattice_vectors(lattice):
    """
    Calculates the reciprocal lattice vectors 
    """

    # B^T*A=2*pi*I <=> B^T = A^(-1)*2*pi*I
    B = (np.linalg.inv(lattice)*2*np.pi).T
    return B


def main():

    file1 = 'dft_chargedensity1.xsf'
    file2 = 'dft_chargedensity2.xsf'

    rho1, lattice1, grid1, shift1 = read_example_xsf_density(file1)
    rho2, lattice2, grid2, shift2 = read_example_xsf_density(file2)

    N_electrons1 = number_of_electrons(rho1,lattice1,grid1)
    N_electrons2 = number_of_electrons(rho2,lattice2,grid2)

    print("Number of electrons:")
    print("Structure 1:", "{:.4f}".format(N_electrons1))
    print("Structure 2:", "{:.4f}".format(N_electrons2))
    print("")

    B1 = reciprocal_lattice_vectors(lattice1)
    B2 = reciprocal_lattice_vectors(lattice2)

    print("Reciprocal lattice vectors:")
    print("Structure 1:\n", B1)
    print("Structure 2:\n", B2)
    
if __name__=="__main__":
    main()



