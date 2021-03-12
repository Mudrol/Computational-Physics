#! /usr/bin/env python3

"""
FYS-4096 Computational Physics, Exercise 7, problem 4

Based on the hartree_1d.py file

"""



from numpy import *
from matplotlib.pyplot import *
import matplotlib.patches as mpatches
from numpy.lib.polynomial import _polyfit_dispatcher
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps
import h5py
import os

def hartree_potential(ns,x):
    """ 
    Hartree potential using Simpson integration 

    Args:
        ns: Densities
        x: gridpoints

    Returns:
        Vhartree: Hartree potential
    """

    # Initialize the potential
    Vhartree=0.0*ns

    # Loop through every gridpoint
    for ix in range(len(x)):
        r = x[ix]

        # Initialize function to be integrated
        f = 0.0*x

        # For every gridpoint, calculate the value of the integrand
        for ix2 in range(len(x)):
            rp = x[ix2]
            f[ix2]=ns[ix2]*ee_potential(r-rp)
        
        # Integrate using simps
        Vhartree[ix]=simps(f,x)
    return Vhartree

def ee_potential(x):
    global ee_coef
    """ 1D electron-electron interaction """
    return ee_coef[0]/sqrt(x**2+ee_coef[1])

def ext_potential(x,m=1.0,omega=1.0):
    """ 1D harmonic quantum dot """
    return 0.5*m*omega**2*x**2

def density(psis):
    """
    Calculates the electron density given as
    n(x) = sum_i |psi_i(x)|^2, in other words, for every coordinate point,
    sum the psis of the different orbitals.
    """
    # Convert to numpy array
    psis = np.array(psis)

    # sum column-wise (axis=0)
    ns = sum(np.abs(psis)**2,axis=0)
    return ns
    
def initialize_density(x,dx,normalization=1):
    """ Initialize the density and normalize using simps """
    rho=exp(-x**2)
    A=simps(rho,x)
    return normalization/A*rho

def check_convergence(Vold,Vnew,threshold):
    """ Check if the potentials are within the threshold """
    difference_ = amax(abs(Vold-Vnew))
    print('  Convergence check:', difference_)
    converged=False
    if difference_ <threshold:
        converged=True
    return converged

def diagonal_energy(T,orbitals,x):
    """ 
    Calculate diagonal energy
    (using Simpson)
    """
    Tt=sp.csr_matrix(T)
    E_diag=0.0
    
    for i in range(len(orbitals)):
        evec=orbitals[i]
        E_diag+=simps(evec.conj()*Tt.dot(evec),x)
    return E_diag

def offdiag_potential_energy(orbitals,x):
    """ 
    Calculate off-diagonal energy
    (using Simpson)
    """
    U = 0.0
    for i in range(len(orbitals)-1):
        for j in range(i+1,len(orbitals)):
            fi = 0.0*x
            for i1 in range(len(x)):
                fj = 0.0*x
                for j1 in range(len(x)):
                    fj[j1]=abs(orbitals[i][i1])**2*abs(orbitals[j][j1])**2*ee_potential(x[i1]-x[j1])
                fi[i1]=simps(fj,x)
            U+=simps(fi,x)
    return U

def save_data_to_hdf5_file(fname,ns,N_e,occ,grid,ee_coefs,Ek,Ep,Et):
    """ Save the data into a hdf5 file """

    # Create hdf5 file
    hf = h5py.File(fname+'.hdf5','w')

    # Add the datasets
    hf.create_dataset('ns', data=ns,dtype='f')
    hf.create_dataset('N_e', data=N_e,dtype='i8')
    hf.create_dataset('occ', data=occ,dtype='i8')
    hf.create_dataset('grid', data=grid,dtype='f')
    hf.create_dataset('ee_coefs', data=ee_coefs,dtype='f')
    hf.create_dataset('E_kin', data=Ek,dtype='f')
    hf.create_dataset('E_pot', data=Ep,dtype='f')
    hf.create_dataset('E_tot', data=Et,dtype='f')

    hf.close()

def calculate_SIC(orbitals,x):
    """ Calculate the self interaction correction for V_eff """
    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(-hartree_potential(abs(orbitals[i])**2,x))
    return V_SIC
            
def normalize_orbital(evec,x):
    """ Normalize orbitals to one """
    
    # Normalize orbital by first calculating the density, then integrating
    # and finally getting the square root to get the inverse of normalization constant
    A = np.sqrt(simps(np.abs(evec)**2,x))
    evec_norm = 1./A * evec
    return evec_norm
 
def kinetic_hamiltonian(x):
    """ Returns the kinetic hamiltonian """
    grid_size = x.shape[0]
    dx = x[1] - x[0]
    dx2 = dx**2
    
    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])
    return H0

def hartree_1d(x,dx,N_e,occ,ee_coef,T,Vext,maxiters,mix,threshold):
    """
    Solves the system using Hartree approximation, SCF method. Callable function
    to enable solving for different parameters

    Based on the hartree_1d.py file's main function.
    """
    
    ns=initialize_density(x,dx,N_e)

    print('Density integral        ', simps(ns,x))
    print(' -- should be close to  ', N_e)
    print('\nCalculating initial state')
    Vhartree=hartree_potential(ns,x)
    VSIC=[]
    for i in range(N_e):
        # Now VSIC is initialized as zero, since there are no orbitals yet
        VSIC.append(ns*0.0)

    Veff=sp.diags(Vext+Vhartree,0)
    H=T+Veff

    iterations = 0
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        iterations += 1
        orbitals=[]
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i+1)
            eigs, evecs = sla.eigs(H+sp.diags(VSIC[i],0), k=N_e, which='SR')
            eigs=real(eigs)
            evecs=real(evecs)
            print('    eigenvalues', eigs)
            evecs[:,occ[i]]=normalize_orbital(evecs[:,occ[i]],x)
            orbitals.append(evecs[:,occ[i]])
        Veff_old = 1.0*Veff
        ns=density(orbitals)
        Vhartree=hartree_potential(ns,x)
        VSIC=calculate_SIC(orbitals,x)
        Veff_new=sp.diags(Vext+Vhartree,0)
        if check_convergence(Veff_old,Veff_new,threshold):
            break
        else:
            """ Mixing the potential """
            Veff= mix*Veff_old+(1-mix)*Veff_new
            H = T+Veff

    print('\n\n')
    off = offdiag_potential_energy(orbitals,x)
    E_kin = diagonal_energy(T,orbitals,x)
    E_pot = diagonal_energy(sp.diags(Vext,0),orbitals,x) + off
    E_tot = E_kin + E_pot

    return ns,E_kin,E_pot,E_tot,iterations

def main():

     # --- Setting up the system etc. ---
    global ee_coef
    # e-e potential parameters [strenght, smoothness], 'a and b'
    ee_coef = [1.0, 1.0]

    # number of electrons
    N_e = 6

    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up
    occ_0 = [0,0,1,1,2,2] # Ground state
    occ_3 = [0,0,1,2,3,3] # S=3: spin down from 2 excited and both spins from 2 excited
    occs = [occ_0, occ_3] # For creating for loop
    
    # grid
    x=linspace(-4,4,120)

    # threshold
    threshold=1.0e-4

    # mixing value
    mix=0.2

    # maximum number of iterations
    maxiters = 100

    # Initialization and calculations start here
    dx = x[1]-x[0]
    T = kinetic_hamiltonian(x)
    Vext = ext_potential(x)

    # Arrays for storing data from the calculations
    E_pots = []
    E_kins = []
    E_tots = []
    states_ns = []
    states_iters = [] # Amount of SCF iterations

    # Solve the system for both Hartree states (interacting)
    for occ in occs:
        density, E_kin, E_pot, E_tot, iters = hartree_1d(x,dx,N_e,occ,ee_coef,
                                                  T,Vext,maxiters,mix,threshold)
        states_ns.append(density)
        E_kins.append(E_kin)
        E_pots.append(E_pot)
        E_tots.append(E_tot)
        states_iters.append(iters)

    # Save the data into hdf5 files
    save_data_to_hdf5_file('problem4_data_0_int',states_ns[0],N_e,occ_0,x,ee_coef,
                           E_kins[0],E_pots[0],E_tots[0])
    save_data_to_hdf5_file('problem4_data_3_int',states_ns[1],N_e,occ_3,x,ee_coef,
                           E_kins[1],E_pots[1],E_tots[1])    


    # Change into non-interacting case (a=0)
    ee_coef = [0, 1.0]
    for occ in occs:
        density, E_kin, E_pot, E_tot, iters = hartree_1d(x,dx,N_e,occ,ee_coef,
                                                  T,Vext,maxiters,mix,threshold)
        states_ns.append(density)
        E_kins.append(E_kin)
        E_pots.append(E_pot)
        E_tots.append(E_tot)
        states_iters.append(iters)

    # Print the energies for both states, and also amount of iterations
    print('----------INTERACTING CASE----------\n')
    for i in range(2):
        print('Total energy for state S={}:      {} '.format(3*i ,E_tots[i]))
        print('Kinetic energy for state S={}:    {} '.format(3*i ,E_kins[i]))
        print('Potential energy for state S={}:  {} '.format(3*i ,E_pots[i]))
        print('Iterations needed for S={}:       {}'.format(3*i, states_iters[i]))
        print('')
    
    # Print same things for the non-interacting case
    print('--------NON-INTERACTING CASE--------\n')
    for i in range(2,4):
        print('Total energy for state S={}:      {} '.format(3*(i-2) ,E_tots[i]))
        print('Kinetic energy for state S={}:    {} '.format(3*(i-2) ,E_kins[i]))
        print('Potential energy for state S={}:  {} '.format(3*(i-2) ,E_pots[i]))
        print('Iterations needed for S={}:       {}'.format(3*(i-2), states_iters[i]))
        print('')
    
    # Comparing the energies and printing them
    print('--------COMPARING INTERACTING AND NON-INTERACTING CASE-------\n')
    for i in range(2):
        print('Total energy difference for state S={}:     {}'.format(3*i,
                                                    abs(E_tots[i]-E_tots[i+2])))
        print('Kinetic energy difference for state S={}:   {}'.format(3*i,
                                                    abs(E_kins[i]-E_kins[i+2])))
        print('Potential energy difference for state S={}: {}'.format(3*i,
                                                    abs(E_pots[i]-E_pots[i+2])))
        print('')

    # Save data in hdf5
    save_data_to_hdf5_file('problem4_data_0_noint',states_ns[2],N_e,occ_0,x,ee_coef,
                           E_kins[2],E_pots[2],E_tots[2])
    save_data_to_hdf5_file('problem4_data_3_noint',states_ns[3],N_e,occ_0,x,ee_coef,
                           E_kins[3],E_pots[3],E_tots[3])

    dens_0, = plot(x,abs(states_ns[0]),label='S = 0')
    dens_3, = plot(x,abs(states_ns[1]),label='S = 3')
    xlabel(r'$x$ (a.u.)')
    ylabel(r'$n(x)$ (1/a.u.)')
    title('N-electron density for N={0}'.format(N_e))

    # Add the energies into legend using mpatches
    kinstr1 = r'$E_{k,0} = $' + '{:.2f}'.format(E_kins[0])
    kinstr2 = r'$E_{k,3} = $' + '{:.2f}'.format(E_kins[1])
    potstr1 = r'$E_{p,0} = $' + '{:.2f}'.format(E_pots[0])
    potstr2 = r'$E_{p,3} = $' + '{:.2f}'.format(E_pots[1])
    totstr1 =  r'$E_{tot,0} = $' + '{:.2f}'.format(E_tots[0])
    totstr2 =  r'$E_{tot,3} = $' + '{:.2f}'.format(E_tots[1])
    text1 = mpatches.Patch(color='white',label=kinstr1)
    text11 = mpatches.Patch(color='white',label=kinstr2)
    text2 = mpatches.Patch(color='white',label=potstr1)
    text22 = mpatches.Patch(color='white',label=potstr2)
    text3 = mpatches.Patch(color='white',label=totstr1)
    text33 = mpatches.Patch(color='white',label=totstr2)

    legend(handles=[dens_0,dens_3,text1,text11,text2,text22,text3,text33])
    show()

if __name__=="__main__":
    main()