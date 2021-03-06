"""
    FYS-4096 Computational Physics: Project Work 1.

    Problem 5: Stationary state Schrödinger equation in 2D.
    This program solves the single particle Schrödinger equation in 2D using
    FDM. The first two energies are calculated and the ground state is plotted
    along the analytical answer.

    In this version, the potential is perturbated by a Gaussian perturbation.
    
    H*psi(x,y) = E*psi(x,y)

    Made by: Matias Hiillos
"""
import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.integrate import simps
from scipy.sparse.linalg import eigsh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, PercentFormatter


def fdm_2d(N,L,x,y,h,k):
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

        
    # potential coordinates
    pot_x = np.repeat(x**2,N)
    pot_y = np.tile(y**2,N)
    # Add Gaussian perturbation
    # f(x) = a*exp(-(x-b)^2/(2*c^2)), a,b real constants, c non zero constant
    # gauss(x,y) = f(x)*f(y)

    # Perturbation indeces
    pert_area = np.array([int(N*0.5),int(N*0.65)])
    pot_x = pot_x.reshape((N,N))
    pot_y = pot_y.reshape((N,N))

    # Add perturbation to the area specified by the indeces
    a = 10
    b = -5
    c = 100
    for i in range(pert_area[0],pert_area[1]):
        for j in range(pert_area[0],pert_area[1]):
            pot_y[i][j] = pot_y[i][j] + gauss_fun(np.sqrt(pot_y[i][j]),a,b,c)
            pot_x[i][j] = pot_x[i][j] + gauss_fun(np.sqrt(pot_x[i][j]),a,b,c)
    
    # reshape back to 1d arrays for the Hamiltonian matrix
    pot_x_1d = pot_x.flatten()
    pot_y_1d = pot_y.flatten()

    # Plot the new potential
    X,Y = np.meshgrid(x,y)

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.plot_surface(X, Y, pot_x+pot_y, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax = fig.add_subplot(1,2,2)
    fig.suptitle(r'Potential with a Gaussian perturbation')
    ax.imshow(pot_x+pot_y,extent=[-L/2,L/2,-L/2,L/2])
    plt.savefig('perturbated_potential.png')
    plt.close()


    # The whole Hamiltonian in matrix form
    A = (-1*lap + sps.diags(pot_x_1d) + sps.diags(pot_y_1d))/2

    # Calculate the 10 smallest eigenvalues and corresponding eigenvector
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

def gauss_fun(x,a,b,c):
    """Gaussian function used for creating perturbation.
       Parameters used for Gaussian function are a=100, b=0, c=1.
       -> f(x) = 100*exp(-x^2/2)

    Args:
        x : coordinate

    Returns:
        function value at x
    """

    return a*np.exp(-(x-b)**2/(2*c))

def normalized_density(psi,x):
    """Normalizes a single state, and returns the energy 
       density of the normalized state

    Args:
        psi: Single state as an 1D array

    Returns:
        The normalized density of the state
    """
    
    # Calculate the density, then integrate to get the normalization constant
    psi = np.abs(psi)**2
    A = simps(simps(psi,x),x)
    A = 1./A


    # Normalize the density
    psi = psi * A

    return psi

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


def main():
    """Main function for initalizing the system and handling the printing and 
       plotting
    """
    N = 200  # Amount of gridpoints
    L = 10  # Size of the system
    k = 50
    x = y = np.linspace(-L/2,L/2,N)
    h = x[1]-x[0]

    E,psi = fdm_2d(N,L,x,y,h,k)

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
    print("TESTI:", simps(simps(psi00_exact_density,x),x))

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
    plt.savefig('FDM_psi00_perturbated.png')
    plt.close()


    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig2.add_subplot(1,2,1,projection='3d')
    surf2 = ax.plot_surface(X, Y, psi00_exact_density, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax = fig2.add_subplot(1,2,2)
    fig2.suptitle(r'Analytical normalized ground state density $|\psi|^2$')
    ax.imshow(psi00_exact_density,extent=[-L/2,L/2,-L/2,L/2])
    plt.savefig('exact_psi00_perturbated.png')
    plt.close()

    # Plot some of the other densities and save them as pdf
    for i in range(1,20):
        state = densities_norm[i]
        fig = plt.figure(figsize=plt.figaspect(0.5))
        plt.imshow(state,extent=[-L/2,L/2,-L/2,L/2])
        plt.title('n={}'.format(i))
        plt.savefig('FDM_perturbated{}.png'.format(i))
        plt.close()

    # Plot the energies of analytical and perturbated case

    # Get analytical energies from nx,ny = 0 to 10
    n = 10
    energies = analytical_energies(n)

    # Plot k analytical and the FDM energies
    index = np.arange(k)
    plt.figure()
    plt.plot(index,energies[0:k],label='Analytical energies')
    plt.plot(index,E,label='Energies of the perturbated system')
    plt.legend()
    plt.title('Energies')
    plt.xlabel('n')
    plt.ylabel(r'$\tilde{E} = \frac{E}{\hbar\omega}$')
    plt.savefig('energies_perturbation.png')
    plt.close()

if __name__=="__main__":
    main()
