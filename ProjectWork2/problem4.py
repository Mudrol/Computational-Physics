"""
FYS-4096 Computational Physics, Project work 2, problem 4:
2D Ising model of a ferromagnet

This file simulates the Ising model of a ferromagnet in a regular 2D grid using
Monte Carlo. Simulations are done in two ways with varying temperatures: 
    a) Nearest neighbor interactions only, J1 = 4
    b) Nearest and next nearest neighbor interactions, J1 = 4, J2 = 1

For the nearest neighbor case, phase transition temperature was at around 
2.2-2.4. By including the next nearest neighbors into the calculations the phase
transition temperature rose to around 2.9-3.2.

Due to the finite size of the simulated system, calculating the small system
includes systematic errors called finite size effects. The simulations get more
accurate by increasing the size, but to get the 'natural' phase transition, the
size of the system should be infinite. This is why additionally for the case
a) we also vary the size of the grid with values [4,8,16,32,64]. As the system
grows bigger, we need to find new equilibration points and if the equilibration
of the blocks happens too late, we also have to increase the amount of blocks
calculated. By increasing the N we can see that the energy per spin after the
equilibrium point saturates much better with smaller variance to the mean value.

Note: On the figures, the magnetic susceptibility at N=64 shoots to higher numbers
compared to the others. However, magnetization wasn't studied in this simulation
as much compared to the energy so it is hard to say why exactly it is giving a 
higher value.


Based on spin_mc_simple.py file
"""


from numpy import *
from matplotlib.pyplot import *
import matplotlib.patches as mpatches
from scipy.special import erf

class Walker:
    def __init__(self,*args,**kwargs):
        self.spin = kwargs['spin']
        self.nearest_neighbors = kwargs['nn']
        self.next_nearest_neighbors = kwargs['nnn']
        self.sys_dim = kwargs['dim']
        self.coords = kwargs['coords']

    def w_copy(self):
        return Walker(spin=self.spin.copy(),
                      nn=self.nearest_neighbors.copy(),
                      nnn = self.next_nearest_neighbors.copy(),
                      dim=self.sys_dim,
                      coords=self.coords.copy())
    

def Energy(Walkers):
    """ Returns energy of a microstate s """
    E = 0.0
    
    # Nearest neighbor
    for walker in Walkers:
        E += site_Energy(Walkers,walker)

    # Every nn-pair is iterated twice
    return E*0.5


def site_Energy(Walkers,Walker):
    E = 0.0
    J1 = 4.0 #units of k_B
    J2 = 0 # 0: only nn, 1: nn and next-nn

    # interaction between nn
    for k in range(len(Walker.nearest_neighbors)):
        j = Walker.nearest_neighbors[k]
        E += -J1*Walker.spin*Walkers[j].spin

    # Interaction between next-nn as desired
    if J2 > 0:
        for k in range(len(Walker.next_nearest_neighbors)):
            j = Walker.next_nearest_neighbors[k]
            E += -J2*Walker.spin*Walkers[j].spin
    
    return E


def magnetic_moment(Walkers):
    """ Returns the magnetic moment of microstate s """
    mag = 0.0
    for walker in Walkers:
        mag += walker.spin
    return mag


def ising(Nblocks,Niters,Walkers,beta):
    M = len(Walkers)
    Eb = zeros((Nblocks,))
    Ebsq = zeros((Nblocks,))
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))
    mag = zeros((Nblocks,))
    magsq = zeros((Nblocks,))

    obs_interval = 5
    for i in range(Nblocks):
        EbCount = 0
        for j in range(Niters):
            site = int(random.rand()*M)

            s_old = 1.0*Walkers[site].spin
            E_old = site_Energy(Walkers,Walkers[site])

            #coinflip for flipping spin
            s_new = random.randint(2) - 0.5
            Walkers[site].spin = 1.0*s_new

            E_new = site_Energy(Walkers,Walkers[site])
            deltaE = E_new-E_old

            # Metropolis Monte Carlo
            # Spin only flips between two states (uniform distribution),
            # so only the difference of energies of these microstates needs to be measured
            q_s_sp = exp(-beta*(deltaE))
            if (q_s_sp > random.rand()):
                    Accept[i] += 1.0
            else:
                Walkers[site].spin = s_old
            AccCount[i] += 1
            if j % obs_interval == 0:
                # not calculating per spin
                E_tot = Energy(Walkers)
                mag_moment = magnetic_moment(Walkers)
                mag[i] += mag_moment
                magsq[i] += mag_moment**2
                Eb[i] += E_tot
                Ebsq[i] += E_tot**2
                EbCount += 1
            
        Eb[i] /= EbCount
        mag[i] /= EbCount
        Ebsq[i] /= EbCount
        magsq[i] /= EbCount

        Accept[i] /= AccCount[i]

        """
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))
        """

    return Walkers, Eb, Accept, mag, Ebsq, magsq


def main():

    # lattice
    dim = 2
    grids = [4,8,16,32,64] # N

    # Average values per spin for each N
    Es = []
    Evars = []
    mags = []
    magvars = []
    msuss = []
    Cvs = []

    # Simulate the system with different N
    for grid_side in grids:
        Walkers=[]
        grid_size = grid_side**dim

        mapping = zeros((grid_side,grid_side),dtype=int) # mapping
        inv_map = [] # inverse mapping
        ii = 0
        for i in range(grid_side):
            for j in range(grid_side):
                mapping[i,j]=ii
                inv_map.append([i,j])
                ii += 1
    

        # Create walkers
        for i in range(grid_side):
            for j in range(grid_side):
                # Calculate the coordinates of nearest neighbors, 
                # periodic boundary conditions apply
                j1=mapping[i,(j-1) % grid_side]
                j2=mapping[i,(j+1) % grid_side]
                i1=mapping[(i-1) % grid_side,j]
                i2=mapping[(i+1) % grid_side,j]

                # Same for next-nn
                nnn1=mapping[(i-1) % grid_side,(j-1) % grid_side]
                nnn2=mapping[(i-1) % grid_side,(j+1) % grid_side]
                nnn3=mapping[(i+1) % grid_side,(j-1) % grid_side]
                nnn4=mapping[(i+1) % grid_side,(j+1) % grid_side]
                Walkers.append(Walker(spin=0.5,
                                    nn=[j1,j2,i1,i2],
                                    nnn=[nnn1,nnn2,nnn3,nnn4],
                                    dim=dim,
                                    coords = [i,j]))
        Niters = 1000

        # Calculate more blocks for larger N
        if grid_side > 50:
            Nblocks = 500
        else:
            Nblocks = 100

        # Change equilibration time based on N
        if grid_side == 64:
            eq = 100
        else:
            eq = 20
        T = 3
        beta = 1.0/T
        """
        Notice: Energy is measured in units of k_B, which is why
                beta = 1/T instead of 1/(k_B T)
        """
        Walkers, Eb, Acc, mag, Ebsq, magsq = ising(Nblocks,Niters,Walkers,beta)

        # Plot the energy per spin for a given N
        figure()
        plot(Eb/grid_size)
        plot([eq,eq],gca().get_ylim(),'k--')
        plot(gca().get_xlim(),[mean(Eb[eq:])/grid_size,mean(Eb[eq:])/grid_size],'r--',label='mean = {:.4f}'.format(mean(Eb[eq:])/grid_size))
        title('Block energies for N={}'.format(grid_side))
        xlabel('n')
        ylabel('E')
        legend()
        savefig('Block_energies_N_{}.png'.format(grid_side))

        mag = mag[eq:]
        magsq = magsq[eq:]
        Eb = Eb[eq:]
        Ebsq = Ebsq[eq:]

        mag_per_spin = mean(mag)/grid_size
        mag_var_per_spin = std(mag)/grid_size/sqrt(len(mag))
        E_per_spin = mean(Eb)/grid_size
        E_var_per_spin = std(Eb)/sqrt(len(Eb))/grid_size
        msus_per_spin = (mean(magsq)-mean(mag)**2)/grid_size/T
        Cv_per_spin = (mean(Ebsq)-mean(Eb)**2)/grid_size/T**2

        Es.append(E_per_spin)
        mags.append(mag_per_spin)
        Evars.append(E_var_per_spin)
        magvars.append(mag_var_per_spin)
        msuss.append(msus_per_spin)
        Cvs.append(Cv_per_spin)



        #plot(Eb)
        print('N =',grid_side)
        print('Ising total energy: {0:.5f} +/- {1:0.5f}'.format(E_per_spin, E_var_per_spin))
        print('Variance to energy ratio: {0:.5f}'.format(abs(var(Eb)/mean(Eb)))) 

        print('Heat capacity: {0:.5f}'.format(Cv_per_spin))
        print('Magnetization: {0:.5f}'.format(mag_per_spin))
        print('Magnetic susceptibility: {0:.5f}'.format(msus_per_spin))

    ind = arange(len(grids))
    # Plot the results
    figure()
    bar(ind,Es,yerr=Evars)
    xlabel('N')
    ylabel('E')
    title('Energy per spin')
    xticks(ind,grids)
    savefig('finitesize_energy.png')

    figure()
    bar(ind,mags,yerr=magvars)
    xlabel('N')
    ylabel('M')
    title('Magnetization per spin')
    xticks(ind,grids)
    savefig('finitesize_mag.png')

    figure()
    bar(ind,Cvs)
    xlabel('N')
    ylabel(r'$C_V$')
    title('Heat capacity per spin')
    xticks(ind,grids)
    savefig('finitesize_Cv.png')

    figure()
    bar(ind,msuss)
    xlabel('N')
    ylabel(r'$\chi$')
    title('Magnetic susceptibility per spin')
    xticks(ind,grids)
    savefig('finitesize_mag_sus.png')

def ising_temperature():
    """ Performs the Monte Carlo simulations for temperature in the range [0.5,6] """

    Walkers=[]
    dim = 2
    grid_side = 10
    grid_size = grid_side**dim

    mapping = zeros((grid_side,grid_side),dtype=int) # mapping
    inv_map = [] # inverse mapping
    ii = 0
    for i in range(grid_side):
        for j in range(grid_side):
            mapping[i,j]=ii
            inv_map.append([i,j])
            ii += 1
    

    # Create walkers
    for i in range(grid_side):
        for j in range(grid_side):
            # Calculate the coordinates of nearest neighbors, 
            # periodic boundary conditions apply
            j1=mapping[i,(j-1) % grid_side]
            j2=mapping[i,(j+1) % grid_side]
            i1=mapping[(i-1) % grid_side,j]
            i2=mapping[(i+1) % grid_side,j]

            # Same for next-nn
            nnn1=mapping[(i-1) % grid_side,(j-1) % grid_side]
            nnn2=mapping[(i-1) % grid_side,(j+1) % grid_side]
            nnn3=mapping[(i+1) % grid_side,(j-1) % grid_side]
            nnn4=mapping[(i+1) % grid_side,(j+1) % grid_side]
            Walkers.append(Walker(spin=0.5,
                                nn=[j1,j2,i1,i2],
                                nnn=[nnn1,nnn2,nnn3,nnn4],
                                dim=dim,
                                coords = [i,j]))
    
        
    Nblocks = 200
    Niters = 1000
    eq = 20 # equilibration "time"
    sims = 20 # Temperature points
    temps = linspace(0.5,6.0,sims)
    Cvs = zeros(sims)
    Es = zeros(sims)
    Es_std = zeros(sims)
    magns = zeros(sims)
    magns_std = zeros(sims)
    magn_suss = zeros(sims)
    i = 0
    for temp in temps:
        beta = 1.0/temp
        # Change back to initial system: all spin up
        for walker in Walkers:
            walker.spin = 0.5
        # Solve the system
        Walkers, Eb, Acc, mag, Ebsq, magsq = ising(Nblocks,Niters,Walkers,beta)
        
        mag = mag[eq:]
        magsq = magsq[eq:]
        Eb = Eb[eq:]
        Ebsq = Ebsq[eq:]

        E_pspin = mean(Eb)/grid_size
        mag_pspin = mean(mag)/grid_size
        Cv_pspin = (mean(Ebsq)-mean(Eb)**2)/grid_size/temp**2
        msus_pspin = (mean(magsq)-mean(mag)**2)/grid_size/temp

        Es[i] = E_pspin
        Es_std[i] = std(Eb)/sqrt(len(Eb))/grid_size
        magns_std[i] = std(mag)/len(mag)/grid_size
        Cvs[i] = Cv_pspin
        magns[i] = mag_pspin
        magn_suss[i] = msus_pspin
        
        i+=1

    # Phase transition line
    x = [3.1,3.1]
    
    # Plotting
    
    figure()
    errorbar(temps,Es,yerr = Es_std)
    xlabel('T')
    ylabel('Energy per spin, NNN')
    savefig('nnn_energy_per_spin.png')
    figure()
    plot(temps,Cvs,'o-')
    plot(x,gca().get_ylim(),'k--')
    xlabel('T')
    ylabel('Heat capacity per spin, NNN')
    savefig('nnn_Cv_per_spin.png')
    figure()
    errorbar(temps,magns,yerr=magns_std)
    xlabel('T')
    ylabel('Magnetization per spin, NNN')
    savefig('nnn_magnetization_per_spin.png')
    figure()
    plot(temps,magn_suss,'o-')
    xlabel('T')
    ylabel('Susceptibility per spin, NNN')
    plot(x,gca().get_ylim(),'k--')
    savefig('nnn_susceptibility_per_spin.png')


    show()

if __name__=="__main__":
    main() # For the simulation with varying grid sizes
    #ising_temperature() # For performing the simulations with varying temperatures
        
