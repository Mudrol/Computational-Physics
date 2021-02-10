"""
    FYS-4096 Computational Physics: Exercise 5
    Problem 3: Poisson equation and relaxation methods

    This program solves the Poisson equation in 2d using the Jacobi,
    Gauss-Seidel and SOR update schemes. Grid spacing for both directions are
    same.

    Made by: Matias Hiillos
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

eps_0 = 8.854*1e-11     # Vaccuum permittivity

def jacobi(phi, h, rho, N, tol=1e-4, count=0):
    """
    Jacobi update scheme. Program calls itself recursively until tolerance is
    reached.
    
    Parameters:
    phi: function to be updated
    h: grid spacing
    rho: charge density
    N: amount of gridpoints (NxN)
    tol: tolerance
    count: counter for amount of updates to reach tolerance

    Returns:
    phi2: Updated function
    count: the total count of updates done

    """
    phi2 = phi.copy()

    # Periodic boundary condition: Ignore first and last indices
    for i in range(1,N-1):
        for j in range(1,N-1):
            phi2[i,j] = 1/4 * (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1] + h**2/eps_0*rho[i,j])

    # Check that the updated phi is within tolerance, in other words the updated
    # value differs less than the tolerance value
    if np.max(np.abs(phi-phi2)) > tol:
         return jacobi(phi2,h,rho,N,count=count+1)
    else: 
        return phi2, count

def gs(phi, h, rho, N, tol=1e-4, count=0):
    """
    Gauss-Seidel update scheme Program calls itself recursively until tolerance is
    reached.

    Parameters:
    phi: function to be updated
    h: grid spacing
    rho: charge density
    N: amount of gridpoints (NxN)
    tol: tolerance
    count: counter for amount of updates to reach tolerance

    Returns:
    phi2: Updated function
    count: the total count of updates done
    """
    limit = 950
    phi2 = phi.copy()

    # Periodic boundary condition: Ignore first and last indices
    for i in range(1,N-1):
        for j in range(1,N-1):
            phi2[i,j] = 1/4 * (phi[i+1,j] + phi2[i-1,j] + phi[i,j+1] + phi2[i,j-1] + h**2/eps_0*rho[i,j])
    
    # Check that the updated phi is within tolerance, in other words the updated
    # value differs less than the tolerance value
    if np.max(np.abs(phi-phi2)) > tol and count < limit:
         return gs(phi2,h,rho,N,count=count+1)
    else: 
        return phi2, count

def sor(phi, h, rho, N, w=1.8, tol=1e-4, count=0):
    """
    SOR update scheme. Program calls itself recursively until tolerance is
    reached.

    Parameters:
    phi: function to be updated
    h: grid spacing
    rho: charge density
    N: amount of gridpoints (NxN)
    w: omega, Value between [1,2] used in the algorithm
    tol: tolerance
    count: counter for amount of updates to reach tolerance

    Returns:
    phi2: Updated function
    count: the total count of updates done
    """
    phi2 = phi.copy()

    # Periodic boundary condition: Ignore first and last indices
    for i in range(1,N-1):
        for j in range(1,N-1):
            phi2[i,j] = (1-w)*phi[i,j] + w/4 * (phi[i+1,j] + phi2[i-1,j] + phi[i,j+1] + phi2[i,j-1] + h**2*rho[i,j]/eps_0)
    
    # Check that the updated phi is within tolerance, in other words the updated
    # value differs less than the tolerance value
    if np.max(np.abs(phi-phi2)) > tol:
         return sor(phi2,h,rho,N,count=count+1)
    else: 
        return phi2, count


def main():
    
    L = 1
    N = 20
    h = 1/N

    phi = np.zeros((N,N))

    # Boundary conditions
    phi[:,0] = 1
    phi_jacobi = phi_gs = phi_sor = phi

    rho = np.zeros((N,N))
    x = np.linspace(0,L,N)
    y = np.linspace(0,L,N)
    X,Y = np.meshgrid(x,y)

    phi_jacobi, n_jacobi = jacobi(phi_jacobi, h, rho, N)
    phi_gs, n_gs = gs(phi_jacobi, h, rho, N)
    phi_sor, n_sor = sor(phi_jacobi, h, rho, N)

    # Print how many iterations were needed for the same accuracy
    print("Updates needed for accuracy of 1e-4:")
    print("Jacobi:",n_jacobi)
    print("Gauss-Seidel:",n_gs)
    print("SOR:",n_sor)

    # Plotting
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.plot_wireframe(X,Y,phi_jacobi)
    ax1.text2D(0., 0.95, "Jacobi method", transform=ax1.transAxes)

    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    ax2.plot_wireframe(X,Y,phi_gs)
    ax2.text2D(0., 0.95, "Gauss-Seidel method", transform=ax2.transAxes)

    fig3 = plt.figure()
    ax3 = Axes3D(fig3)
    ax3.plot_wireframe(X,Y,phi_sor)
    ax3.text2D(0., 0.95, "SOR method", transform=ax3.transAxes)

    plt.show()





if __name__=="__main__":
    main()