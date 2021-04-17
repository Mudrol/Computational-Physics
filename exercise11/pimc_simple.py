"""
    FYS-4096 Computational Physics: Exercise 11, problem 4.
    modified version of pimc_simple.py to simulate two hydrogen atoms in 3D.
    Interaction is simulated by Morse potential energy surface

    Trotter numbers for classical: M=1, for qm: M=8 and M=16, Temperature T=300K
"""

from numpy import *
from matplotlib.pyplot import *
import matplotlib.patches as mpatches
from scipy.special import erf

class Walker:
    def __init__(self,*args,**kwargs):
        self.Ne = kwargs['Ne']
        self.Re = kwargs['Re']
        self.spins = kwargs['spins']
        self.Nn = kwargs['Nn']
        self.Rn = kwargs['Rn']
        self.Zn = kwargs['Zn']
        self.tau = kwargs['tau']
        self.sys_dim = kwargs['dim']

    def w_copy(self):
        return Walker(Ne=self.Ne,
                      Re=self.Re.copy(),
                      spins=self.spins.copy(),
                      Nn=self.Nn,
                      Rn=self.Rn.copy(),
                      Zn=self.Zn,
                      tau=self.tau,
                      dim=self.sys_dim)
    

def kinetic_action(r1,r2,tau,lambda1):
    return sum((r1-r2)**2)/lambda1/tau/4

def potential_action(Walkers,time_slice1,time_slice2,tau):
    return 0.5*tau*(potential(Walkers[time_slice1]) \
                    +potential(Walkers[time_slice2]))

def pimc(Nblocks,Niters,Walkers):
    M = len(Walkers)
    
    # proton and electron masses in atomic units
    mp = 1836
    me = 1
    Ne = Walkers[0].Ne*1
    sys_dim = 1*Walkers[0].sys_dim
    tau = 1.0*Walkers[0].tau
    # Hydrogen: one electron and one proton -> m = mp+me (electron could probably be excluded)
    lambda1 = 0.5/(mp+me)
    Eb = zeros((Nblocks,))
    # Internuclear distances
    ri = zeros((Nblocks,))
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))
    sigma2 = lambda1*tau
    sigma = sqrt(sigma2)

    obs_interval = 5
    for i in range(Nblocks):
        EbCount = 0
        for j in range(Niters):
            time_slice0 = int(random.rand()*M)
            time_slice1 = (time_slice0+1)%M
            time_slice2 = (time_slice1+1)%M
            ptcl_index = int(random.rand()*Ne)

            r0 = Walkers[time_slice0].Re[ptcl_index]
            r1 = 1.0*Walkers[time_slice1].Re[ptcl_index]
            r2 = Walkers[time_slice2].Re[ptcl_index]
 
            KineticActionOld = kinetic_action(r0,r1,tau,lambda1) +\
                kinetic_action(r1,r2,tau,lambda1)
            PotentialActionOld = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)

            # bisection sampling
            r02_ave = (r0+r2)/2
            log_S_Rp_R = -sum((r1-r02_ave)**2)/2/sigma2             
            Rp = r02_ave + random.randn(sys_dim)*sigma
            log_S_R_Rp = -sum((Rp - r02_ave)**2)/2/sigma2

            Walkers[time_slice1].Re[ptcl_index] = 1.0*Rp
            KineticActionNew = kinetic_action(r0,Rp,tau,lambda1) +\
                kinetic_action(Rp,r2,tau,lambda1)
            PotentialActionNew = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)

            deltaK = KineticActionNew-KineticActionOld
            deltaU = PotentialActionNew-PotentialActionOld
            #print('delta K', deltaK)
            #print('delta logS', log_S_R_Rp-log_S_Rp_R)
            #print('exp(dS-dK)', exp(log_S_Rp_R-log_S_R_Rp-deltaK))
            #print('deltaU', deltaU)
            q_R_Rp = exp(log_S_Rp_R-log_S_R_Rp-deltaK-deltaU)
            A_RtoRp = min(1.0,q_R_Rp)
            if (A_RtoRp > random.rand()):
                Accept[i] += 1.0
            else:
                Walkers[time_slice1].Re[ptcl_index]=1.0*r1
            AccCount[i] += 1
            if j % obs_interval == 0:
                E_kin, E_pot = Energy(Walkers)
                #print(E_kin,E_pot)
                Eb[i] += E_kin + E_pot
                EbCount += 1

                # Internuclear distance, r1 is the one moved
                r = sqrt(sum((Walkers[time_slice1].Re[0]-Walkers[time_slice1].Re[1])**2))
                ri[i] += r
            #exit()
            
        Eb[i] /= EbCount
        ri[i] /= EbCount
        Accept[i] /= AccCount[i]
        #print('Block {0}/{1}'.format(i+1,Nblocks))
        #print('    E   = {0:.5f}'.format(Eb[i]))
        #print('    Acc = {0:.5f}'.format(Accept[i]))


    return Walkers, Eb, Accept, ri


def Energy(Walkers):
    M = len(Walkers)
    mp = 1836
    me = 1
    d = 1.0*Walkers[0].sys_dim
    tau = Walkers[0].tau
    lambda1 = 0.5/(mp+me)
    U = 0.0
    K = 0.0
    for i in range(M):
        U += potential(Walkers[i])
        for j in range(Walkers[i].Ne):
            if (i<M-1):
                K += d/2/tau-sum((Walkers[i].Re[j]-Walkers[i+1].Re[j])**2)/4/lambda1/tau**2
            else:
                K += d/2/tau-sum((Walkers[i].Re[j]-Walkers[0].Re[j])**2)/4/lambda1/tau**2    
    return K/M,U/M
        
    

def potential(Walker):
    V = 0.0
    r_cut = 1.0e-12

    # Morse parameters for H2 in atomic units
    re = 1.4 #a0
    De = 0.1745 # Ha
    a = 1.0282 #a0^-1

    # Morse potential
    for i in range(Walker.Ne-1):
        for j in range(i+1,Walker.Ne):
            r = sqrt(sum((Walker.Re[i]-Walker.Re[j])**2))
            V += De*(1-exp(-a*(r-re)))**2-De

    
    return V

def main():

    # Trotter numbers, M=1: classical MC
    M = [1,8,16]

    T = 300 # Temperature K
    kb = 3.16681e-6 # Boltzmann constant Ha/K
   
    Ebs = []
    ris = []
    Emeans = []
    rimeans = []
    for m in M:
        Walkers = []
        tau = 1/T/m/kb
        # For H2
        # Note: 'electrons' in this code are actually protons
        for i in range(m):
            Walkers.append(Walker(Ne=2,
                                Re=[array([-0.7,0,0]),array([0.7,0,0])],
                                spins=[0,1],
                                Nn=2,
                                Rn=[array([-0.7,0,0]),array([0.7,0,0])],
                                Zn=[1.0,1.0],
                                tau = tau,
                                dim=3))
        
        Nblocks = 200
        Niters = 100
        
        Walkers, Eb, Acc, ri = pimc(Nblocks,Niters,Walkers)
        Ebs.append(Eb)
        ris.append(ri)
        #plot(Eb)
        Eb = Eb[10:]
        ri = ri[10:]
        Emeans.append(mean(Eb))
        rimeans.append(mean(ri))

        print('M = {}:'.format(m))
        print('PIMC total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb), std(Eb)/sqrt(len(Eb))))
        print('Variance to energy ratio: {0:.5f}'.format(abs(var(Eb)/mean(Eb))))
        print('Average internuclear distance: {0:.5f}'.format(mean(ri)))
        print('')
    
    
    figure(figsize=(8,6))
    cut = 10
    x = [range(200)]
    plthandls = []
    for i in range(len(M)):
        plthandls.append(scatter(x,Ebs[i],s=5,label='M={}'.format(M[i])))
    plot([cut,cut],gca().get_ylim(),'k--')

    M1E = r'$\langle E_{M=1} \rangle$ = ' + '{:.4f} Ha'.format(Emeans[0])
    M8E = r'$\langle E_{M=8} \rangle$ = ' + '{:.4f} Ha'.format(Emeans[1])
    M16E =  r'$\langle E_{M=16} \rangle$ = ' + '{:.4f} Ha'.format(Emeans[2])

    text1 = mpatches.Patch(color='white',label=M1E)
    text2 = mpatches.Patch(color='white',label=M8E)
    text3 = mpatches.Patch(color='white',label=M16E)

    legend(handles=[plthandls[0],plthandls[1],plthandls[2],text1,text2,text3])
    xlabel('n')
    ylabel('E (Ha)')
    title('Comparison of energies for classical and quantum atoms, M = [1,8,16]')
    savefig('problem4_E_comparison.png')

    figure(figsize=(8,6))
    plthandls = []
    for i in range(len(M)):
        plthandls.append(scatter(x,ris[i],s=5,label='M={}'.format(M[i])))
    plot([cut,cut],gca().get_ylim(),'k--')

    M1r = r'$\langle r_{M=1} \rangle$ = ' + r'{:.4f} $a_0$'.format(rimeans[0])
    M8r = r'$\langle r_{M=8} \rangle$ = ' + r'{:.4f} $a_0$'.format(rimeans[1])
    M16r =  r'$\langle r_{M=16} \rangle$ = ' + r'{:.4f} $a_0$'.format(rimeans[2])

    text1 = mpatches.Patch(color='white',label=M1r)
    text2 = mpatches.Patch(color='white',label=M8r)
    text3 = mpatches.Patch(color='white',label=M16r)
    legend(handles=[plthandls[0],plthandls[1],plthandls[2],text1,text2,text3])
    xlabel('n')
    ylabel(r'r ($a_0$)')
    title('Comparison of internuclear distances for classical and quantum atoms, M = [1,8,16]')
    savefig('problem4_r_comparison.png')

    #show()
if __name__=="__main__":
    main()
        
