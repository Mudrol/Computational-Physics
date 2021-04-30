"""
FYS-4096 Computational Physics, Project work 2, problem 4:
2D Ising model of a ferromagnet

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
    J2 = 0 # 0: only nn, 2: nn and next-nn

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
    Walkers=[]

    # 10x10 lattice
    dim = 2
    grid_side = 10
    grid_size = grid_side**dim # Also amount of spins
    
    # Ising model nearest neighbors only
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
    T = 3
    beta = 1.0/T
    """
    Notice: Energy is measured in units of k_B, which is why
            beta = 1/T instead of 1/(k_B T)
    """
    Walkers, Eb, Acc, mag, Ebsq, magsq = ising(Nblocks,Niters,Walkers,beta)


    mag = mag[eq:]
    magsq = magsq[eq:]
    Eb = Eb[eq:]
    Ebsq = Ebsq[eq:]
    Cv = (Ebsq-Eb**2)/T**2
    msus = (magsq-mag**2)/T


    #plot(Eb)
    print('Ising total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb)/grid_size, std(Eb)/sqrt(len(Eb))/grid_size))
    print('Variance to energy ratio: {0:.5f}'.format(abs(var(Eb)/mean(Eb)))) 

    print('Heat capacity: {0:.5f}'.format((mean(Ebsq)-mean(Eb)**2)/T**2/grid_size))
    print('Magnetization: {0:.5f}'.format(mean(mag)/grid_size))
    print('Magnetic susceptibility: {0:.5f}'.format((mean(magsq)-mean(mag)**2)/T/grid_size))


    # Calculate the observables for temperatures in range [0.5, 6.0]
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
        # Solve the system again
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
    x = [2.3, 2.3]

    # Plotting
    
    figure()
    errorbar(temps,Es,yerr = Es_std)
    xlabel('T')
    ylabel('Energy per spin, NN')
    savefig('nn_energy_per_spin.png')
    figure()
    plot(temps,Cvs,'o-')
    plot(x,gca().get_ylim(),'k--')
    xlabel('T')
    ylabel('Heat capacity per spin, NN')
    savefig('nn_Cv_per_spin.png')
    figure()
    errorbar(temps,magns,yerr=magns_std)
    xlabel('T')
    ylabel('Magnetization per spin, NN')
    savefig('nn_magnetization_per_spin.png')
    figure()
    plot(temps,magn_suss,'o-')
    xlabel('T')
    ylabel('Susceptibility per spin, NN')
    plot(x,gca().get_ylim(),'k--')
    savefig('nn_susceptibility_per_spin.png')
    
    show()

if __name__=="__main__":
    main()
        
