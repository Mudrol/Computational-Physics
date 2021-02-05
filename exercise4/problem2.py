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
    # Calculate points of the lattice in alpha space
    alpha0 = np.linspace(0.,1,grid[0])
    alpha1 = np.linspace(0.,1,grid[1])
    alpha2 = np.linspace(0.,1,grid[2])

    # Volume of the lattice
    det_A = np.linalg.det(lattice)

    # Integrate with simps in alpha space for all directions
    # rho(r)dr = rho(A*alpha)*det(A)d(alpha)
    rho_0 = simps(rho[:,0,0],alpha0)
    rho_1 = simps(rho[0,:,0],alpha1)
    rho_2 = simps(rho[0,0,:],alpha2)
    N = det_A*(rho_0*rho_1*rho_2)
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



