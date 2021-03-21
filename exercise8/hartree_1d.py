#! /usr/bin/env python3

"""
Hartree code for N-electron 1D harmonic quantum dot

- Related to FYS-4096 Computational Physics 
- Test case: N=4, occ=[0,0,1,1], m=omega=1 with Hartree approximation
  should give:

    Total energy      11.502873299221452
    Kinetic energy    3.622113606711247
    Potential energy  7.880759692510205

    Density integral  3.9999999999999996

- Job description for Exercise 8 in short:
  - Modify this or your own Hartree code to enable/include 
    Hartree-Fock approximation
  - Notice: you need assign spin variable, and add needed
    functions such as calculation of vix, ni_HF, etc.
"""



from numpy import *
from matplotlib.pyplot import *
import matplotlib.patches as mpatches
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps
import h5py
import os

def hartree_potential(ns,x):
    """ 
    Hartree potential using Simpson integration 
    """
    Vhartree=0.0*ns
    for ix in range(len(x)):
        r = x[ix]
        f = 0.0*x
        for ix2 in range(len(x)):
            rp = x[ix2]
            f[ix2]=ns[ix2]*ee_potential(r-rp)
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
    ns=zeros((len(psis[0]),))
    for i in range(len(psis)):
        ns+=abs(psis[i])**2
    return ns
    
def initialize_density(x,dx,normalization=1):
    rho=exp(-x**2)
    A=simps(rho,x)
    return normalization/A*rho

def check_convergence(Vold,Vnew,threshold):
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

def exchange_energy(orbitals,spins,x):
    """
    Calculate the exchange energy due to introducing of the Pauli principle
    (using Simpson)
    """
    Ex = 0.0
    for i in range(len(orbitals)-1):
        for j in range(i+1,len(orbitals)):
            if spins[i] == spins[j]:
                fi = 0.0*x
                for i1 in range(len(x)):
                    fj = 0.0*x
                    for j1 in range(len(x)):
                        fj[j1] = \
                            orbitals[i][i1] * orbitals[j][j1] \
                          * orbitals[i][j1] * orbitals[j][i1] \
                          * ee_potential(x[i1]-x[j1])
                    fi[i1] = simps(fj,x)

                Ex += -1*simps(fi,x)
    return Ex
                    

def save_ns_in_ascii(ns,filename):
    s=shape(ns)
    f=open(filename+'.txt','w')
    for ix in range(s[0]):
        f.write('{0:12.8f}\n'.format(ns[ix]))
    f.close()
    f=open(filename+'_shape.txt','w')
    f.write('{0:5}'.format(s[0]))
    f.close()
    
def load_ns_from_ascii(filename):
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

def save_data_to_hdf5_file(fname,orbitals,density,N_e,occ,grid,ee_coefs):
    return

def calculate_SIC(orbitals,x):
    V_SIC = []
    for i in range(len(orbitals)):
        V_SIC.append(-hartree_potential(abs(orbitals[i])**2,x))
    return V_SIC
            
def calculate_ex(psis,spins,x):
    """ Calculates the exchange potential for HF """

    V_xs = []

    # Calculate V_x for every orbital, very similar to Hartree potential
    for i in range(len(psis)):
        V_x = 0.*psis[0]
        for ix in range(len(x)):
            r = x[ix]
            f = 0.0*x
            for ix2 in range(len(x)):
                rp = x[ix2]

                # Here get the Hartree-Fock density 
                f[ix2]=hf_density(psis,spins,i,ix,ix2,x)*ee_potential(r-rp)
            V_x[ix]=-1*simps(f,x)
        V_xs.append(V_x)

    return V_xs

def hf_density(psis,spins,ind,rind,rpind,x):
    """Calculates the Hartree-Fock density for single orbital (n_i^{HF})

    Args:
        psis: orbitals
        spins: spins array for checking whether the spins are same
        ind: index of the orbital (phi_i)
        rind: r index
        rpind: rp index
        x: coordinates

    Returns:
        Hartree-Fock density at a point for an orbital specified by index
    """
    n = 0

    for j in range(len(psis)):
        if spins[ind] == spins[j]:
            n += ( psis[j][rpind] * psis[ind][rpind] * psis[j][rind] \
                 / psis[ind][rind] )

    return n


def normalize_orbital(evec,x):
    return evec/sqrt(simps(abs(evec)**2,x))
    #dx = x[1]-x[0]
    #return evec/sqrt((sum(abs(evec)**2)*dx))

def kinetic_hamiltonian(x):
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
    global ee_coef
    # e-e potential parameters [strenght, smoothness]
    ee_coef = [1.0, 1.0]
    # 1D occupations each orbital can have max 2, i.e., spin up and spin down
    # e.g., occ = [0,0,1,1] means lowest up and down, next lowest up and down
    #       occ = [0,1,2,3] means all spin up
    
    #occ = [0,0,1,1,2,2] # P4 S=0
    #occ = [0,1,2,3,4,5] # P4 S=3
    occ = [0,0,1,1,2,2] # S = 0, N = 6

    # For checking if spin is same, in this case spin is alternating
    spins = [1,0,1,0,1,0]

    # number of electrons
    N_e = len(occ)

    # grid
    x=linspace(-4,4,120)
    # threshold
    threshold=1.0e-4
    # mixing value
    mix=0.2
    # maximum number of iterations
    maxiters = 100

    dx = x[1]-x[0]
    T = kinetic_hamiltonian(x)
    Vext = ext_potential(x)

    # READ in density / orbitals / etc.
    if os.path.isfile('density.txt'):
        ns=load_ns_from_ascii('density')
    else:
        ns=initialize_density(x,dx,N_e)

    print('Density integral        ', simps(ns,x))
    print(' -- should be close to  ', N_e)
    
    print('\nCalculating initial state')
    Vhartree=hartree_potential(ns,x)
    VSIC=[]
    Vex=[]
    for i in range(N_e):
        VSIC.append(ns*0.0)
        Vex.append(ns*0.0)
    Veff=sp.diags(Vext+Vhartree,0)
    H=T+Veff

    print('\n Now solving using Hartree method')
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
            Veff=(1.0-mix)*Veff_new+mix*Veff_old
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

    # 'Save' ns for future plotting
    ns_h = ns

    print('\n Now solving using Hartree-Fock method')
    for i in range(maxiters):
        print('\n\nIteration #{0}'.format(i))
        orbitals=[]
        for i in range(N_e):
            print('  Calculating orbitals for electron ', i+1)
            eigs, evecs = sla.eigs(H+sp.diags(Vex[i],0), k=N_e, which='SR')
            eigs=real(eigs)
            evecs=real(evecs)
            print('    eigenvalues', eigs)
            evecs[:,occ[i]]=normalize_orbital(evecs[:,occ[i]],x)
            orbitals.append(evecs[:,occ[i]])
        Veff_old = 1.0*Veff
        ns=density(orbitals)
        Vhartree=hartree_potential(ns,x)
        Vex=calculate_ex(orbitals,spins,x)
        Veff_new=sp.diags(Vext+Vhartree,0)
        if check_convergence(Veff_old,Veff_new,threshold):
            break
        else:
            Veff=(1.0-mix)*Veff_new+mix*Veff_old
            H = T+Veff


    print('\n\n')
    off = offdiag_potential_energy(orbitals,x)
    E_kin_hf = diagonal_energy(T,orbitals,x)
    E_pot_hf = diagonal_energy(sp.diags(Vext,0),orbitals,x) + off + exchange_energy(orbitals,spins,x)
    E_tot_hf = E_kin + E_pot
    print('Total energy     ', E_tot_hf)
    print('Kinetic energy   ', E_kin_hf)
    print('Potential energy ', E_pot_hf) 
    print('\nDensity integral ', simps(ns,x))

    # WRITE OUT density / orbitals / energetics / etc.
    #save_ns_in_ascii(ns,'density')
    
    hart, = plot(x,abs(ns_h),label='Hartree')
    hartfo, = plot(x,abs(ns),label='Hartree-Fock')
    xlabel(r'$x$ (a.u.)')
    ylabel(r'$n(x)$ (1/a.u.)')
    title('N-electron density for N={0}'.format(N_e))

    # Add the energies into legend using mpatches
    kinstr = r'$E_{k,H} = $' + '{:.2f}'.format(E_kin)
    potstr = r'$E_{p,H} = $' + '{:.2f}'.format(E_pot)
    totstr =  r'$E_{tot,H} = $' + '{:.2f}'.format(E_tot)
    kinstr_hf = r'$E_{k,HF} = $' + '{:.2f}'.format(E_kin_hf)
    potstr_hf = r'$E_{p,HF} = $' + '{:.2f}'.format(E_pot_hf)
    totstr_hf =  r'$E_{tot,HF} = $' + '{:.2f}'.format(E_tot_hf)
    text1 = mpatches.Patch(color='white',label=kinstr)
    text2 = mpatches.Patch(color='white',label=potstr)
    text3 = mpatches.Patch(color='white',label=totstr)
    text1_hf = mpatches.Patch(color='white',label=kinstr_hf)
    text2_hf = mpatches.Patch(color='white',label=potstr_hf)
    text3_hf = mpatches.Patch(color='white',label=totstr_hf)
    legend(handles=[hart,hartfo,text1,text2,text3,text1_hf,text2_hf,text3_hf])

    savefig('problem1.png')

if __name__=="__main__":
    main()
