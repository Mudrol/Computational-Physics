"""
Simple Monte Carlo for Ising model

Related to course FYS-4096 Computational Physics

Problem 1:
- Make the code to work, that is, include code to where it reads "# ADD"
- Comment the parts with "# ADD" and make any additional comments you 
  think could be useful for yourself later.
- Follow the assignment from ex11.pdf.

Problem 2:
- Add observables: heat capacity, magnetization, magnetic susceptibility
- Follow the assignment from ex11.pdf.

Problem 3:
- Look at the temperature effects and locate the phase transition temperature.
- Follow the assignment from ex11.pdf.

Phase transition happens around T = 2.5K

"""


from numpy import *
from matplotlib.pyplot import *
from scipy.special import erf

class Walker:
    def __init__(self,*args,**kwargs):
        self.spin = kwargs['spin']
        self.nearest_neighbors = kwargs['nn']
        self.sys_dim = kwargs['dim']
        self.coords = kwargs['coords']

    def w_copy(self):
        return Walker(spin=self.spin.copy(),
                      nn=self.nearest_neighbors.copy(),
                      dim=self.sys_dim,
                      coords=self.coords.copy())
    

def Energy(Walkers):
    """ Returns energy of a microstate s """
    E = 0.0
    J = 4.0 # given in units of k_B
    # Simplest Ising model assumption: interaction only between nearest neighbors.
    # Energy of such assumption: E(s) = -J*sum(si*sj), summing over all nn pairs
    for walker in Walkers:
        E += site_Energy(Walkers,walker)

    # Every nn-pair is iterated twice
    return E*0.5


def site_Energy(Walkers,Walker):
    E = 0.0
    J = 4.0 # given in units of k_B
    # Simple Ising model, interaction only between nn
    for k in range(len(Walker.nearest_neighbors)):
        j = Walker.nearest_neighbors[k]
        E += -J*Walker.spin*Walkers[j].spin
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
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))
    mag = zeros((Nblocks,))

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
                E_tot = Energy(Walkers)/M # energy per spin
                mag[i] = abs(magnetic_moment(Walkers))/M # magn. moment per spin
                Eb[i] += E_tot
                EbCount += 1
            
        Eb[i] /= EbCount
        Accept[i] /= AccCount[i]

        """
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))
        """

    return Walkers, Eb, Accept, mag


def main():
    Walkers=[]

    dim = 2
    grid_side = 10
    grid_size = grid_side**dim
    
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
            Walkers.append(Walker(spin=0.5,
                                  nn=[j1,j2,i1,i2],
                                  dim=dim,
                                  coords = [i,j]))
 
    
    Nblocks = 200
    Niters = 1000
    eq = 20 # equilibration "time"
    T = 3.0
    beta = 1.0/T
    """
    Notice: Energy is measured in units of k_B, which is why
            beta = 1/T instead of 1/(k_B T)
    """
    Walkers, Eb, Acc, mag = ising(Nblocks,Niters,Walkers,beta)

    #plot(Eb)
    Eb = Eb[eq:]
    print('Ising total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb), std(Eb)/sqrt(len(Eb))))
    print('Variance to energy ratio: {0:.5f}'.format(abs(var(Eb)/mean(Eb)))) 

    # TODO: Fix scaling for magnetic susceptibility and heat capacity
    # Cv (should be) now in units of kb
    print('Heat capacity: {0:.5f}'.format((mean(abs(Eb)**2)-mean(abs(Eb))**2)/T**2)) # add scaling factor
    print('Magnetization: {0:.5f}'.format(mean(mag)))
    print('Magnetic susceptibility: {0:.5f}'.format((mean(abs(mag)**2)-mean(abs(mag))**2)/T)) # add scaling factor

    # Now Calculate the observables for temperatures in range [0.5, 6.0]
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
        Walkers, Eb, Acc, mag = ising(Nblocks,Niters,Walkers,beta)
        
        Eb = Eb[eq:]

        Cv = (mean(abs(Eb)**2)-mean(abs(Eb))**2)/temp**2
        magn_mean = mean(mag)
        magn_sus = ((mean(abs(mag)**2))-(mean(abs(mag))**2))/temp

        Es[i] = mean(Eb)
        Es_std[i] = std(Eb)/sqrt(len(Eb))
        magns_std[i] = std(mag)/len(mag)
        Cvs[i] = Cv
        magns[i] = magn_mean
        magn_suss[i] = magn_sus
        
        i+=1

    # Phase transition line
    x = [2.5, 2.5]

    # Plotting
    figure()
    errorbar(temps,Es,yerr = Es_std)
    xlabel('T')
    ylabel('Energy per spin')
    savefig('problem3_energy_per_spin.png')
    figure()
    plot(temps,Cvs,'o-')
    plot(x,gca().get_ylim(),'k--')
    xlabel('T')
    ylabel('Heat capacity per spin')
    savefig('problem3_Cv_per_spin.png')
    figure()
    errorbar(temps,magns,yerr=magns_std)
    xlabel('T')
    ylabel('Magnetization per spin')
    savefig('problem3_magnetization_per_spin.png')
    figure()
    plot(temps,magn_suss,'o-')
    xlabel('T')
    ylabel('Susceptibility per spin')
    plot(x,gca().get_ylim(),'k--')
    savefig('problem3_susceptibility_per_spin.png')
    #show()

if __name__=="__main__":
    main()
        
