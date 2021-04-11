"""
    FYS-4096 Computational Physics: Exercise 10, Problems 1 & 2

    Problem 1:

    Q: What system did you solve? What is the temperature?

    A: The solved system was the two electron harmonic quantum dot in 2d. The
       temperature can be calculated from the formula tau = 1/(kTM), where
         tau = time-step, 0.25
         k = Boltzmann constant, 3.16681e-6 Ha/K
         T = Temperature, K
         M = Amount of density matrices, 100
        By solving T we get the temperature 
        T = (k*M*tau)^-1 = 12631,007228L = 1,26e+4 K

"""

from numpy import *
from matplotlib.pyplot import *
from scipy.special import erf
import matplotlib.patches as mpatches



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
    """ Kinetic part of the action """
    return sum((r1-r2)**2)/lambda1/tau/4

def potential_action(Walkers,time_slice1,time_slice2,tau):
    """ Potential part of the action """
    return 0.5*tau*(potential(Walkers[time_slice1]) \
                    +potential(Walkers[time_slice2]))

def pimc(Nblocks,Niters,Walkers):
    """ Path Integral Monte Carlo """
    M = len(Walkers)
    Ne = Walkers[0].Ne*1
    sys_dim = 1*Walkers[0].sys_dim
    tau = 1.0*Walkers[0].tau
    lambda1 = 0.5
    Eb = zeros((Nblocks,))
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))
    sigma2 = lambda1*tau
    sigma = sqrt(sigma2)

    obs_interval = 5
    for i in range(Nblocks):
        EbCount = 0
        for j in range(Niters):

            # Get time slices, used to access the three consecutive points of the path
            time_slice0 = int(random.rand()*M)
            time_slice1 = (time_slice0+1)%M
            time_slice2 = (time_slice1+1)%M
            ptcl_index = int(random.rand()*Ne)

            # Locations of walkers
            r0 = Walkers[time_slice0].Re[ptcl_index]
            r1 = 1.0*Walkers[time_slice1].Re[ptcl_index]
            r2 = Walkers[time_slice2].Re[ptcl_index]

            # Calculate the kinetic and potential action
            KineticActionOld = kinetic_action(r0,r1,tau,lambda1) +\
                kinetic_action(r1,r2,tau,lambda1)
            PotentialActionOld = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)

            # bisection sampling / moves
            log_S_Rp_R, Rp, log_S_R_Rp = bisection(r0,r1,r2,sigma,sigma2,sys_dim)

            # uniform sampling
            #log_S_Rp_R, Rp, log_S_R_Rp = uniform(r0,r1,r2,sigma,sigma2,sys_dim)


            Walkers[time_slice1].Re[ptcl_index] = 1.0*Rp

            # New actions from sampling
            KineticActionNew = kinetic_action(r0,Rp,tau,lambda1) +\
                kinetic_action(Rp,r2,tau,lambda1)
            PotentialActionNew = potential_action(Walkers,time_slice0,time_slice1,tau)+potential_action(Walkers,time_slice1,time_slice2,tau)

            # Change in kinetic and potential energy
            deltaK = KineticActionNew-KineticActionOld
            deltaU = PotentialActionNew-PotentialActionOld

            # this could be helpful in checking whether the kinetic
            # part becomes sampled exactly or not (how is that)
            #print('delta K', deltaK)
            #print('delta logS', log_S_R_Rp-log_S_Rp_R)
            #print('exp(dS-dK)', exp(log_S_Rp_R-log_S_R_Rp-deltaK))
            #print('deltaU', deltaU)

            # Trial move and metropolis, q(R->Rp), similar to vmc/dmc in ex.9
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
            #exit()
            
        Eb[i] /= EbCount
        Accept[i] /= AccCount[i]
        print('Block {0}/{1}'.format(i+1,Nblocks))
        print('    E   = {0:.5f}'.format(Eb[i]))
        print('    Acc = {0:.5f}'.format(Accept[i]))


    return Walkers, Eb, Accept

def bisection(r0,r1,r2,sigma,sigma2,sys_dim):
    """ Bisection sampling """
    r02_ave = (r0+r2)/2
    log_S_Rp_R = -sum((r1-r02_ave)**2)/2/sigma2             
    Rp = r02_ave + random.randn(sys_dim)*sigma
    log_S_R_Rp = -sum((Rp - r02_ave)**2)/2/sigma2

    return log_S_Rp_R, Rp, log_S_R_Rp

def uniform(r0,r1,r2,sigma,sigma2,sys_dim):
    """ Uniform sampling """
    r02_ave = (r0+r2)/2
    log_S_Rp_R = -sum((r1-r02_ave)**2)/2/sigma2

    # uniform distribution            
    Rp = r02_ave + random.uniform(-1,1,sys_dim)*sigma
    log_S_R_Rp = -sum((Rp - r02_ave)**2)/2/sigma2

    return log_S_Rp_R, Rp, log_S_R_Rp


def Energy(Walkers):
    """ Calculate the average kinetic and potential energy of the system """
    M = len(Walkers)
    d = 1.0*Walkers[0].sys_dim
    tau = Walkers[0].tau
    lambda1 = 0.5
    U = 0.0
    K = 0.0
    for i in range(M):

        # Potential energy
        U += potential(Walkers[i])
        for j in range(Walkers[i].Ne):

            # Kinetic energy
            if (i<M-1):
                K += d/2/tau-sum((Walkers[i].Re[j]-Walkers[i+1].Re[j])**2)/4/lambda1/tau**2
            else:
                K += d/2/tau-sum((Walkers[i].Re[j]-Walkers[0].Re[j])**2)/4/lambda1/tau**2    

    # Divide by M to get the expectation value (mean)
    return K/M,U/M
        
    

def potential(Walker):
    """ Calculates the potential energy """
    V = 0.0
    r_cut = 1.0e-12
    sigma = 0.05 # smoothing parameter for e-ion interaction
    # e-e
    for i in range(Walker.Ne-1):
        for j in range(i+1,Walker.Ne):
            r = sqrt(sum((Walker.Re[i]-Walker.Re[j])**2))
            V += 1.0/max(r_cut,r)
    
    # For hydrogen: e-ion and ion-ion interactions, modified from ex.9
    if Walker.sys_dim == 3:
        # e-Ion
        for i in range(Walker.Ne):
            for j in range(Walker.Nn):
                r = sqrt(sum((Walker.Re[i]-Walker.Rn[j])**2))
                # 'cumulant' approximation
                V -= Walker.Zn[j]*erf(max(r_cut,r)/sqrt(2*sigma))/max(r_cut,r)

        # Ion-Ion
        for i in range(Walker.Nn-1):
            for j in range(i+1,Walker.Nn):
                r = sqrt(sum((Walker.Rn[i]-Walker.Rn[j])**2))
                V += 1.0/max(r_cut,r)

    Vext = external_potential(Walker)
    
    return V+Vext

def external_potential(Walker):
    """ Calculates the external potential, used for potential energy """
    V = 0.0
    for i in range(Walker.Ne):
        V += 0.5*sum(Walker.Re[i]**2)
        
    return V

def main():
    Walkers=[]

    """
    # For H2
    Walkers.append(Walker(Ne=2,
                          Re=[array([0.5,0,0]),array([-0.5,0,0])],
                          spins=[0,1],
                          Nn=2,
                          Rn=[array([-0.7,0,0]),array([0.7,0,0])],
                          Zn=[1.0,1.0],
                          tau = 0.1,
                          dim=3))
    """

    # For 2D quantum dot
    Walkers.append(Walker(Ne=2,
                          Re=[array([0.5,0]),array([-0.5,0])],
                          spins=[0,1],
                          Nn=2, # not used
                          Rn=[array([-0.7,0]),array([0.7,0])], # not used
                          Zn=[1.0,1.0], # not used
                          tau = 0.25,
                          dim=2))

    M=100
    for i in range(M-1):
         Walkers.append(Walkers[i].w_copy())
    Nblocks = 200
    Niters = 100
    
    Walkers, Eb, Acc = pimc(Nblocks,Niters,Walkers)

    plot(Eb)
    conv_cut=50
    plot([conv_cut,conv_cut],gca().get_ylim(),'k--')
    Eb = Eb[conv_cut:]
    xlabel('m')
    ylabel(r'$\langle E \rangle$ (Ha)')
    title('Calculated energy with PIMC')
    estr = r'Energy mean = {:.3f} Ha'.format(mean(Eb))
    etext = mpatches.Patch(color='white',label=estr)
    legend(handles=[etext])

    print('PIMC total energy: {0:.5f} +/- {1:0.5f}'.format(mean(Eb), std(Eb)/sqrt(len(Eb))))
    print('Variance to energy ratio: {0:.5f}'.format(abs(var(Eb)/mean(Eb)))) 
    show()

if __name__=="__main__":
    main()
        
