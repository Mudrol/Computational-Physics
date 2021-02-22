"""
    FYS-4096 Computational Physics: Project Work 1.

    Problem 5: Stationary state Schrödinger equation in 2D.

    Made by: Matias Hiillos
"""
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh


def fdm_2d():

    N = 51  # Amount of gridpoints
    L = 10  # Size of the system
    x = y = np.linspace(-L/2,L/2,N)
    h = x[1]-x[0]


    # 1d Laplacian, sparse matrix, central difference
    ones = np.ones(N+1)
    diagvalues = np.array([ones,-2*ones,ones])
    offsets = np.array([-1,0,1])
    lap1d = sps.dia_matrix((diagvalues,offsets), shape=(N,N))/h**2
    
    # Represent 2d coordinates as kronecker sum
    lap = sps.kron(lap1d,sps.diags(np.ones(N))) + \
          sps.kron(sps.diags(np.ones(N)),lap1d)

        
    # potential terms
    pot_x = np.repeat(x**2,N)
    pot_y = np.tile(y**2,N)

    A = (-1*lap + sps.diags(pot_x) + sps.diags(pot_y))/2

    E, psi = eigsh(A,k=5,which='SM')
    print(E)

def main():

    fdm_2d()

if __name__=="__main__":
    main()
