"""
    FYS-4096 Computational Physics, Exercise 12, problem 1
    Numerical visualization of the Lennard-Jones potential and the force due
    to the potential for H2.

"""
import numpy as np
import matplotlib.pyplot as plt

def lj_force(r,sig,eps):
    """
    Returns Force due to the LJ-potential, using the formula derived in a)
    """
    return 48*eps*(sig**12/r**13-sig**6/r**7/2)

def lj_potential(r,sig,eps):
    """ 
    Returns Lennard-Jones potential at r, using the formula derived in a)
    """
    return 4*eps*((sig/r)**12-(sig/r)**6)


def main():
    """
        Dissociation energy = eps
        equilibrium distance = rmin
    """
    eps = 0.1745 # Dissociation energy (Ha) (epsilon)
    rmin = 1.4 # Equilibrium distance (potential minimum point), units of a0
    sig = rmin/2**(1/6)

    r = np.linspace(0.3,5,1000)
    U = lj_potential(r,sig,eps)
    F = lj_force(r,sig,eps)
    plt.figure()
    plt.plot(r,U,label='U(r)')
    plt.plot(r,F,label='F(r)')
    plt.xlabel('r')
    plt.title('Lennard-Jones potential and force of H2 molecule')
    plt.ylim(-0.4,0.3)
    plt.xlim(1,3)
    plt.legend()
    plt.savefig('problem1_U_F_plot.png')

if __name__=='__main__':
    main()