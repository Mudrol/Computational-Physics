"""
    FYS-4096 Computational Physics: Exercise 5
    Problem 4: Two capacitor plates

    This program computes the 2D potential profile of two capacitor plates
    inside a square boundary. Also calculates the electric field inside the box.

    Made by: Matias Hiillos
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


eps_0 = 8.854*1e-12     # Vaccuum permittivity

def sor(phi, h, rho, N, w=1.8, tol=1e-4):
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

            # Also dont update at the plates' coordinates (where phi = 1 or -1)
            if np.abs(phi[i,j]) == 1:
                continue
                
            # Update the coordinate according to the formula
            phi2[i,j] = (1-w)*phi[i,j] + w/4 * (phi[i+1,j] + phi2[i-1,j] + phi[i,j+1] + phi2[i,j-1] + h**2*rho[i,j]/eps_0)
    
    # Check that the updated phi is within tolerance, in other words the updated
    # value differs less than the tolerance value
    if np.max(np.abs(phi-phi2)) > tol:
         return sor(phi2,h,rho,N,)
    else: 
        return phi2


def main():
    
    Lp = 1   # Plate length
    L = 2    # Lx, Ly, box lengths
    N = 21   # Grid points, making sure coordinates +-0.3 on x and +-.5 on y included
    h = 1/N  # Grid step length

    phi = np.zeros((N,N))

    # Plate potentials
    phi[7,5:16] = 1
    phi[13,5:16] = -1

    # Charge density on the plates
    rho = np.zeros((N,N))
    rho[7,5:16] = 1
    rho[13,5:16] = -1

    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    X,Y = np.meshgrid(x,y,indexing='ij')

    # Create potential profile using SOR method with omega = 1.8
    sol = sor(phi, h, rho, N)

    # Calculate the electric field, E = -nabla*phi
    Ex,Ey = np.gradient(sol)
    Ex = -Ex
    Ey = -Ey

    # Plotting the potential
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.plot_wireframe(X,Y,sol)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.text2D(0., 0.95, "Potential profile", transform=ax1.transAxes)

    # Plotting the electric field caused by the two plates
    plt.figure()
    plt.quiver(X,Y,Ex,Ey)

    # Plates
    plt.plot([-0.3,-0.3],[-0.5,0.5],'b-', linewidth=3)
    plt.plot([0.3,0.3],[-0.5,0.5],'r-', linewidth=3)

    # Square box
    plt.plot([-1,1],[-1,-1],'k',linewidth=2)
    plt.plot([-1,1],[1,1],'k',linewidth=2)
    plt.plot([-1,-1],[-1,1],'k',linewidth=2)
    plt.plot([1,1],[-1,1],'k',linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Electric field due to two plates, $ E = -\nabla\Phi(x,y)$")

    plt.show()


if __name__=="__main__":
    main()