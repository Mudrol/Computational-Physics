"""
Simple VMC and DMC code for FYS-4096 course at TAU

- Fill in more comments based on lecture slides
- Follow assignment instructions
-- e.g., the internuclear distance should be 1.4 instead of 1.5
   and currently a=1, but it should be 1.1 at first


By Ilkka Kylanpaa


Modifications for determining the improvement of VMC by inclusion of Jastrow factor

Using Jastrow for a poor wave function greatly increases the result of VMC.
"""

from numpy import *
from matplotlib.pyplot import *

class Walker:
    def __init__(self,*args,**kwargs):
        self.Ne = kwargs['Ne']
        self.Re = kwargs['Re']
        self.spins = kwargs['spins']
        self.Nn = kwargs['Nn']
        self.Rn = kwargs['Rn']
        self.Zn = kwargs['Zn']
        self.sys_dim = kwargs['dim']
        self.a = kwargs['a']
        self.useJastrow = kwargs['useJastrow']

    def w_copy(self):
        return Walker(Ne=self.Ne,
                      Re=self.Re.copy(),
                      spins=self.spins.copy(),
                      Nn=self.Nn,
                      Rn=self.Rn.copy(),
                      Zn=self.Zn,
                      dim=self.sys_dim,
                      a = self.a,
                      useJastrow = self.useJastrow)
    

def vmc_run(Nblocks,Niters,Delta,Walkers_in,Ntarget,useJastrow=True):
    """Variational Monte Carlo
    """
    Eb = zeros((Nblocks,))
    Accept=zeros((Nblocks,))

    vmc_walker=Walkers_in[0] # just one walker needed
    Walkers_out=[]
    for i in range(Nblocks):
        for j in range(Niters):
            # moving only electrons
            for k in range(vmc_walker.Ne):
                R = vmc_walker.Re[k].copy()
                Psi_R = wfs(vmc_walker)

                # move the particle
                vmc_walker.Re[k] = R + Delta*(random.rand(vmc_walker.sys_dim)-0.5)
                
                # calculate wave function at the new position
                Psi_Rp = wfs(vmc_walker)

                # calculate the sampling probability
                A_RtoRp = min((Psi_Rp/Psi_R)**2,1.0)
                
                # Metropolis
                if (A_RtoRp > random.rand()):
                    Accept[i] += 1.0/vmc_walker.Ne
                else:
                    vmc_walker.Re[k]=R
                #end if
            #end for
            Eb[i] += E_local(vmc_walker)
        #end for
        if (len(Walkers_out)<Ntarget):
            Walkers_out.append(vmc_walker.w_copy())
        Eb[i] /= Niters
        Accept[i] /= Niters
        #print('Block {0}/{1}'.format(i+1,Nblocks))
        #print('    E   = {0:.5f}'.format(Eb[i]))
        #print('    Acc = {0:.5f}'.format(Accept[i]))
    
    return Walkers_out, Eb, Accept

def H_1s(r1,r2,a):
    return exp(-a*sqrt(sum((r1-r2)**2)))
     
def wfs(Walker):
    """ Calculate the wave function value by including Jastrow function J(R):
        psi(R) = phi(R)*exp(J(R))
    """

    # H2 approx
    f = H_1s(Walker.Re[0],Walker.Rn[0],Walker.a)+H_1s(Walker.Re[0],Walker.Rn[1],Walker.a)
    f *= (H_1s(Walker.Re[1],Walker.Rn[0],Walker.a)+H_1s(Walker.Re[1],Walker.Rn[1],Walker.a))

    if Walker.useJastrow == False:
        return f

    J = 0.0
    # Jastrow e-e
    for i in range(Walker.Ne-1):
        for j in range(i+1,Walker.Ne):
           r = sqrt(sum((Walker.Re[i]-Walker.Re[j])**2))
           if (Walker.spins[i]==Walker.spins[j]):
               J += 0.25*r/(1.0+1.0*r)
           else:
               J += 0.5*r/(1.0+1.0*r)
    
    # Jastrow e-Ion
    for i in range(Walker.Ne):
        for j in range(Walker.Nn):
           r = sqrt(sum((Walker.Re[i]-Walker.Rn[j])**2))
           J -= Walker.Zn[j]*r/(1.0+0.5*r)
       

    return f*exp(J)

def potential(Walker):
    """ Calculates the potential due to electron-electron, ion-ion and 
        electron-ion interactions
    """
    V = 0.0
    r_cut = 1.0e-4
    # e-e
    for i in range(Walker.Ne-1):
        for j in range(i+1,Walker.Ne):
            r = sqrt(sum((Walker.Re[i]-Walker.Re[j])**2))
            V += 1.0/max(r_cut,r)

    # e-Ion
    for i in range(Walker.Ne):
        for j in range(Walker.Nn):
            r = sqrt(sum((Walker.Re[i]-Walker.Rn[j])**2))
            V -= Walker.Zn[j]/max(r_cut,r)

    # Ion-Ion
    for i in range(Walker.Nn-1):
        for j in range(i+1,Walker.Nn):
            r = sqrt(sum((Walker.Rn[i]-Walker.Rn[j])**2))
            V += 1.0/max(r_cut,r)

    return V

def Local_Kinetic(Walker):
    """ Kinetic part of local energy
    """

    # laplacian -0.5 \nabla^2 \Psi / \Psi
    h = 0.001
    h2 = h*h
    K = 0.0
    Psi_R = wfs(Walker)
    for i in range(Walker.Ne):
        for j in range(Walker.sys_dim):
            Y=Walker.Re[i][j]
            Walker.Re[i][j]-=h
            wfs1 = wfs(Walker)
            Walker.Re[i][j]+=2.0*h
            wfs2 = wfs(Walker)
            K -= 0.5*(wfs1+wfs2-2.0*Psi_R)/h2
            Walker.Re[i][j]=Y
    return K/Psi_R

def Gradient(Walker,particle):
    """ Calculates the gradient of a particle """

    h=0.001
    dPsi = zeros(shape=shape(Walker.Re[particle]))
    for i in range(Walker.sys_dim):
        Y=Walker.Re[particle][i]
        Walker.Re[particle][i]-=h
        wfs1=wfs(Walker)
        Walker.Re[particle][i]+=2.0*h
        wfs2=wfs(Walker)
        dPsi[i] = (wfs2-wfs1)/2/h
        Walker.Re[particle][i]=Y

    return dPsi

def E_local(Walker):
    """ Returns the local energy """
    return Local_Kinetic(Walker)+potential(Walker)

def Observable_E(Walkers):
    E=0.0
    Nw = len(Walkers)
    for i in range(Nw):
        E += E_local(Walkers[i])
    E /= Nw
    return E

def main():
    a=0.5
    Walkers=[]
    Walkers.append(Walker(Ne=2,
                        Re=[array([0.5,0,0]),array([-0.5,0,0])],
                        spins=[0,1],
                        Nn=2,
                        Rn=[array([-0.70,0,0]),array([0.70,0,0])], # 1.4a_0
                        Zn=[1.0,1.0],
                        dim=3,
                        a = a,
                        useJastrow = True))
    Ntarget=100
    vmc_time_step = 1.8

    Walkers_jas, Eb_jas, Acc_jas = vmc_run(100,50,vmc_time_step,Walkers,Ntarget)

    # Overwrite Walkers for another VMC calculation
    Walkers=[]
    Walkers.append(Walker(Ne=2,
                        Re=[array([0.5,0,0]),array([-0.5,0,0])],
                        spins=[0,1],
                        Nn=2,
                        Rn=[array([-0.70,0,0]),array([0.70,0,0])], # 1.4a_0
                        Zn=[1.0,1.0],
                        dim=3,
                        a = a,
                        useJastrow = False))

    Walkers, Eb, Acc = vmc_run(100,50,vmc_time_step,Walkers,Ntarget)

    E = mean(Eb)
    E_jas = mean(Eb_jas)
    variance = std(Eb)/sqrt(len(Eb))
    variance_jas = std(Eb_jas)/sqrt(len(Eb_jas))
    vtoe = abs(variance/E)
    vtoe_jas = abs(variance_jas/E_jas)
    print('\n')
    print('Results without Jastrow:')
    print('a =                                 ', a)
    print('Energy:                             ',E)
    print('Variance:                           ',variance)
    print('Absolute variance-to-energy ratio:  ', vtoe)
    print('Average acceptance ratio:           ', mean(Acc))
    print('\n')
    
    print('Results with Jastrow:')
    print('a =                                 ', a)
    print('Energy:                             ',E_jas)
    print('Variance:                           ',variance_jas)
    print('Absolute variance-to-energy ratio:  ', vtoe_jas)
    print('Average acceptance ratio:           ', mean(Acc_jas))

if __name__=="__main__":
    main()
        
