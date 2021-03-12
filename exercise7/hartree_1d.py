#! /usr/bin/env python3

"""
Hartree code for N-electron 1D harmonic quantum dot

- Related to FYS-4096 Computational Physics 
- Test case using simpson integrations should give:

    Total energy      11.502873299221452
    Kinetic energy    3.622113606711247
    Potential energy  7.880759692510205

    Density integral  3.9999999999999996

- Job description in short (more details in the pdf-file): 
  -- Problem 1: 
     - Add/fill needed functions and details, e.g., search for #FILL# 
       and consider whether that is needed for the specific problem.
     - Comment the code and the functions in it. Refer to lecture slides, 
       that is, explain to some extent what is done in the functions
  -- Problem 2: Include input and output as text file
     - some #FILL# sections will give a hint where some input/output 
       functions could be needed
  -- Problem 3: Include input and output as HDF5 file 
     - some #FILL# sections will give a hint where some input/output 
       functions could be needed
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
    sum the psis of the (4) orbitals.
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

def save_ns_in_ascii(ns,filename):
    # Save the densities in ascii file
    s=shape(ns)
    f=open(filename+'.txt','w')
    for ix in range(s[0]):
        f.write('{0:12.8f}\n'.format(ns[ix]))
    f.close()
    f=open(filename+'_shape.txt','w')
    f.write('{0:5}'.format(s[0]))
    f.close()
    
def load_ns_from_ascii(filename):
    # Load the densities from an ascii file
    f=open(filename+'_shape.txt','r')
    for line in f:
        s=array(line.split(),dtype=int)
    f.close()
    ns=zeros((s[0],))
    d=loadtxt(filename+'.txt')
    k=0
    for ix in range(s[0]):
        ns[ix]=d[k]
        k+=1
    return ns

def save_data_in_ascii(arr,filename):
    """ Save data into text file """ 

    # Because orbitals is multidimensional, concatenate them into one list while saving 
    # the length of arrays
    if filename == 'orbitals':
        savetxt(filename+'.txt',concatenate(arr,axis=0))

        # save length of each array in the list
        with open(filename+'_len.txt','w') as f:
            f.write(str(len(arr[0])))

    # Write integer data into file
    elif isinstance(arr, (int,float)):
        with open(filename+'.txt', 'w') as f:
            f.write(str(arr))

    # Write numpy arrays into file
    else:
        savetxt(filename+'.txt',arr)


def load_density_from_ascii(filename):
    """ Read densities from text file """

    # Read the orbitals
    if os.path.exists(filename+'_len.txt'):
        fl = open(filename+'_len.txt','r')
        s = int(fl.readline())
        fl.close()

        dens = []
        # For every orbital TODO: N_e instead of 4
        f = open(filename+'.txt','r')
        for i in range(4):
            # Initialize orbital
            arr = np.zeros(s)
            for j in range(s):
                arr[j] = f.readline()
            dens.append(arr)
        f.close()
        return dens

def load_data_from_ascii(filename):
    """ Read the other data (Excluding density and ns) """
    
    if filename == 'N_e':
        f = open(filename+'.txt')
        N_e = int(f.readline())
        f.close()
        return N_e
    else:
        arr = np.loadtxt(filename+'.txt')
        return arr

def save_data_to_hdf5_file(fname,orbitals,ns,N_e,occ,grid,ee_coefs):
    """ Save the data into a hdf5 file """

    # Create hdf5 file
    hf = h5py.File(fname+'.hdf5','w')

    # Add the datasets
    hf.create_dataset('orbitals', data=orbitals,dtype='f')
    hf.create_dataset('ns', data=ns,dtype='f')
    hf.create_dataset('N_e', data=N_e,dtype='i8')
    hf.create_dataset('occ', data=occ,dtype='i8')
    hf.create_dataset('grid', data=grid,dtype='f')
    hf.create_dataset('ee_coefs', data=ee_coefs,dtype='f')
    hf.close()

def load_data_from_hdf5(fname):
    """ Load the data from an existing hdf5 file """

    # Open file
    hf = h5py.File(fname+'.hdf5','r')

    # Convert from dataset into numpy array
    orbitals = np.array(hf.get('orbitals'))
    ns = np.array(hf.get('ns'))
    occ = np.array(hf.get('occ'))
    grid = np.array(hf.get('grid'))
    ee_coefs = np.array(hf.get('ee_coefs'))

    # Get N_e as scalar since it's not an array
    N_e = asscalar(array(hf.get('N_e')))

    return orbitals,ns,N_e,occ,grid,ee_coefs

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

def main():

     # --- Setting up the system etc. ---
    global ee_coef
    # e-e potential parameters [strenght, smoothness]
    ee_coef = [1.0, 1.0]

    # number of electrons
    N_e = 4

    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up
    occ = [0,1,2,3]

    # grid
    x=linspace(-4,4,120)

    # threshold
    threshold=1.0e-4

    # mixing value
    mix=0.2

    # maximum number of iterations
    maxiters = 100
    # --- End setting up the system etc. ---



    # Initialization and calculations start here
    dx = x[1]-x[0]
    T = kinetic_hamiltonian(x)
    Vext = ext_potential(x)

    """
    #FILL#
    In problems 2 and 3: READ in density / orbitals / etc.
    """


    if os.path.isfile('density.txt'):
        ns=load_ns_from_ascii('density')
    else:
        ns=initialize_density(x,dx,N_e)

    """
    HDF5 file and text file reading, the ASCII part can be commented out to test
    hdf5. Both have been tested. 
    """

    # If there exists a HDF5 file, read the data and add flag for further checks
    dataFile = False
    if os.path.isfile('data.h5'):

        orbitals,ns,N_e,occ,x,ee_coef = load_data_from_hdf5('data')

        # flag is true
        dataFile = True

        # Also increase accuracy
        threshold = 1.0e-10

    # Ascii, now if orbitals text file exists it also assumes that the others exist
    if os.path.isfile('orbitals.txt'):
        orbitals = load_density_from_ascii('orbitals')
        N_e = load_data_from_ascii('N_e')
        occ = load_data_from_ascii('occ').astype(int)
        x = load_data_from_ascii('grid')
        ee_coef = load_data_from_ascii('ee_coef')
        
        dataFile = True
        threshold = 1.0e-10


    print('Density integral        ', simps(ns,x))
    print(' -- should be close to  ', N_e)
    
    print('\nCalculating initial state')
    Vhartree=hartree_potential(ns,x)
    VSIC=[]
    for i in range(N_e):
        # Now VSIC is initialized as zero, since there are no orbitals yet
        VSIC.append(ns*0.0)
 
        """
          #FILL#
          In problems 2 and 3 this needs to be modified, since 
          then you have orbitals already at this point !!!!!!!!!!!!
        """
    # calculate SIC if the datafile exists and has been read
    if dataFile:

        #hdf5
        VSIC = calculate_SIC(orbitals,x)


    Veff=sp.diags(Vext+Vhartree,0)
    H=T+Veff
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
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
    print('Total energy     ', E_tot)
    print('Kinetic energy   ', E_kin)
    print('Potential energy ', E_pot) 
    print('\nDensity integral ', simps(ns,x))

    """
    #FILL#
    In problems 2 and 3:
    WRITE OUT to files: density / orbitals / energetics / etc.
    save_ns_in_ascii(ns,'density') etc.
    """

    # Save data into text files
    save_ns_in_ascii(ns,'density')
    save_data_in_ascii(orbitals,'orbitals')
    save_data_in_ascii(N_e,'N_e')
    save_data_in_ascii(occ,'occ')
    save_data_in_ascii(x,'grid')
    save_data_in_ascii(ee_coef,'ee_coef')

    # Save data in hdf5
    save_data_to_hdf5_file('data',orbitals,ns,N_e,occ,x,ee_coef)

    dens, = plot(x,abs(ns),label='density')
    xlabel(r'$x$ (a.u.)')
    ylabel(r'$n(x)$ (1/a.u.)')
    title('N-electron density for N={0}'.format(N_e))

    # Add the energies into legend using mpatches
    kinstr = r'$E_{k} = $' + '{:.2f}'.format(E_kin)
    potstr = r'$E_{p} = $' + '{:.2f}'.format(E_pot)
    totstr =  r'$E_{tot} = $' + '{:.2f}'.format(E_tot)
    text1 = mpatches.Patch(color='white',label=kinstr)
    text2 = mpatches.Patch(color='white',label=potstr)
    text3 = mpatches.Patch(color='white',label=totstr)

    legend(handles=[dens,text1,text2,text3])
    show()

if __name__=="__main__":
    main()
