"""
    FYS-4096 Computational Physics: Project Work 1.

    Problem 5: Stationary state Schrödinger equation in 2D.
    This program solves the single particle Schrödinger equation in 2D using
    FDM. The first two energies are calculated and the ground state is plotted
    along the analytical answer.
    
    H*psi(x,y) = E*psi(x,y)

    Made by: Matias Hiillos
"""
import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def fdm_2d(N,L,x,y,h):
    """Solves the 2d HO problem

    Args:
        N: amount of gridpoints
        L: length of the system
        x: x-coordinates
        y: y-coordinates
        h: grid spacing

    Returns:
        E: Eigenvalues as a vector
        psi: Eigenvectors corresponding to the eigenvalues in the shape of [:,i]
    """

    # 1d Laplacian, sparse matrix, central difference
    ones = np.ones(N)
    diagvalues = np.array([ones,-2*ones,ones])
    offsets = np.array([-1,0,1])
    lap1d = sps.dia_matrix((diagvalues,offsets), shape=(N,N))/h**2
    
    # Represent 2d coordinates as kronecker sum
    lap = sps.kron(lap1d,sps.diags(np.ones(N))) + \
          sps.kron(sps.diags(np.ones(N)),lap1d)

        
    # potential terms
    pot_x = np.repeat(x**2,N)
    pot_y = np.tile(y**2,N)

    # The whole Hamiltonian in matrix form
    A = (-1*lap + sps.diags(pot_x) + sps.diags(pot_y))/2

    # Calculate the two smallest eigenvalues and corresponding eigenvector
    E, psi = eigsh(A,k=2,which='SM')

    return E,psi

def phi(x,n):
    """Function used for analytical solution of the wave function

    Args:
        x = datapoints
        n = quantum number
    """
    return 1/(np.sqrt(2**n*np.math.factorial(n))) * (1/np.pi)**0.25 * \
           np.exp(-x**2/2)*sp.special.eval_hermite(n,x)

def main():
    """Main function for initalizing the system and handling the printing and 
       plotting
    """
    N = 200  # Amount of gridpoints
    L = 10  # Size of the system
    x = y = np.linspace(-L/2,L/2,N)
    h = x[1]-x[0]

    E,psi = fdm_2d(N,L,x,y,h)

    # Printing energies and the absolute error of the energies
    print('Energies of the two lowest states:')
    print('E_00 = %.4f' % E[0])
    print('E_01 = %.4f' % E[1], '\n')
    print('Absolute error for E_00: %.4e' % np.abs(E[0]-1))
    print('Absolute error for E_01: %.4e' % np.abs(E[1]-2))

    # Normalize the ground-state wavevector
    psi00_normalized = psi[:,0] / np.sqrt(np.sum(psi[:,0]**2))

    # Reshape into 'meshgrid' form
    psi00_normalized = np.reshape(psi00_normalized,(N,N))

    # Density of the state
    psi00_density = psi00_normalized**2

    # Analytical solution of the ground state
    X,Y = np.meshgrid(x,y)
    psi00_exact = phi(X,0)*phi(Y,0)
    psi00_normalized = psi00_exact / np.sqrt(np.sum(psi00_exact**2))
    psi00_exact_density = psi00_normalized**2

    print('\nAbsolute maximum error of the normalized ground state density:')
    print('errmax = %.4e' % np.max(np.abs(psi00_exact_density-psi00_density)))

    # Plotting
    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    surf1 = ax.plot_surface(X, Y, psi00_density, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')
    surf2 = ax.plot_surface(X, Y, psi00_exact_density, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()


if __name__=="__main__":
    main()
