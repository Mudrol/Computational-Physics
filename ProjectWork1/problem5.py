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
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def fdm_2d(N,x,y,h,k):
    """Solves the 2d HO problem

    Args:
        N: amount of gridpoints
        L: length of the system
        x: x-coordinates
        y: y-coordinates
        h: grid spacing
        k: Amount of eigenvalues to be calculated

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
    E, psi = eigsh(A,k=k,which='SM')

    return E,psi

def phi(x,n):
    """Function used for analytical solution of the wave function

    Args:
        x = datapoints
        n = quantum number
    """
    return 1/(np.sqrt(2**n*np.math.factorial(n))) * (1/np.pi)**0.25 * \
           np.exp(-x**2/2)*sp.special.eval_hermite(n,x)

def energy(nx,ny):
    """Calculates the analytical energy 

    Args:
        nx: quantum number of x-direction
        ny: quantum number of y-direction

    Returns:
        the energy
    """
    return 1+nx+ny

def analytical_energies(n):
    """Calculates all of the energies from nx,ny ranging from 0 to n

    Args:
        n: maximum value of the quantum numbers

    Returns:
        Energies sorted as a numpy array
    """

    energies = []
    for nx in range(n):
        for ny in range(n):
            energies.append(energy(nx,ny))
    energies = np.sort(energies)
    return energies

def normalized_density(psi,x):
    """Normalizes a single state, and returns the energy density

    Args:
        psi: Single state as an 1D array
    """
    
    # Calculate the density, then integrate to get the normalization constant
    psi = np.abs(psi)**2
    A = simps(simps(psi,x),x)
    A = 1./A


    # Normalize the density
    psi = psi * A

    return psi

def main():
    """Main function for initalizing the system and handling the printing and 
       plotting
    """
    N = 200  # Amount of gridpoints
    L = 10  # Size of the system
    k = 50   # Amount of energies and states calculated
    x = y = np.linspace(-L/2,L/2,N)
    h = x[1]-x[0]


    E,psi = fdm_2d(N,x,y,h,k)

    # Printing energies and the absolute error of the energies
    print('Energies of the two lowest states:')
    print('E_00 = %.4f' % E[0])
    print('E_01 = %.4f' % E[1], '\n')
    print('Absolute error for E_00: %.4e' % np.abs(E[0]-1))
    print('Absolute error for E_01: %.4e' % np.abs(E[1]-2))

    # Calculate the normalized densities of the states
    densities_norm = np.zeros((k,N,N))
    i = 0
    for state in psi.T:
        # meshgrid form
        state = np.reshape(state,(N,N))
        densities_norm[i] = normalized_density(state,x)
        i += 1

    # Analytical solution of the ground state
    X,Y = np.meshgrid(x,y)
    psi00_exact = phi(X,0)*phi(Y,0)
    psi00_exact_density = normalized_density(psi00_exact,x)

    print('\nMaximum absolute error of the normalized ground state densities:')
    print('errmax = {:.4e}'.format(np.max(np.abs(densities_norm[0]-psi00_exact_density))))

    # Plotting the ground state density
    fig1 = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig1.add_subplot(1,2,1,projection='3d')
    surf1 = ax.plot_surface(X, Y, densities_norm[0], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig1.suptitle(r'Normalized ground state density $|\psi|^2$ using FDM')
    ax = fig1.add_subplot(1,2,2)
    ax.imshow(densities_norm[0],extent=[-L/2,L/2,-L/2,L/2])
    plt.savefig('FDM_psi00_unperturbated.png')
    plt.close()


    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig2.add_subplot(1,2,1,projection='3d')
    surf2 = ax.plot_surface(X, Y, psi00_exact_density, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Plot some of the other densities and save them as pdf
    for i in range(1,20):
        density = densities_norm[i]
        fig = plt.figure(figsize=plt.figaspect(0.5))
        plt.imshow(density,extent=[-L/2,L/2,-L/2,L/2])
        plt.title('n={}'.format(i))
        plt.savefig('FDM_unperturbated{}.png'.format(i))
        plt.close()

    # Plot analytical states until nx,ny = 5
    for nx in range(6):
        for ny in range(6):
            state = phi(X,nx)*phi(Y,ny)
            density = normalized_density(state,x)
            plt.figure()
            plt.imshow(density,extent=[-L/2,L/2,-L/2,L/2])
            plt.title('$n_x={}, n_y={}$'.format(nx,ny))
            plt.savefig('analytical_state_{}_{}.png'.format(nx,ny))
            plt.close()

    # Get analytical energies from nx,ny = 0 to 10
    n = 10
    energies = analytical_energies(n)

    # Plot k analytical and the FDM energies
    index = np.arange(k)
    plt.figure()
    plt.plot(index,energies[0:k],label='Analytical energies')
    plt.plot(index,E,label='FDM energies')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel(r'$\tilde{E} = \frac{E}{\hbar\omega}$')
    plt.title('Energies')
    plt.savefig('energies_unperturbated.png')
    plt.close()


if __name__=="__main__":
    main()
