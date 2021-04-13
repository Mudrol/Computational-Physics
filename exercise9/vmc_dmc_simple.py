"""
Simple VMC and DMC code for FYS-4096 course at TAU

- Fill in more comments based on lecture slides
- Follow assignment instructions
-- e.g., the internuclear distance should be 1.4 instead of 1.5
   and currently a=1, but it should be 1.1 at first


By Ilkka Kylanpaa
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

    def w_copy(self):
        return Walker(Ne=self.Ne,
                      Re=self.Re.copy(),
                      spins=self.spins.copy(),
                      Nn=self.Nn,
                      Rn=self.Rn.copy(),
                      Zn=self.Zn,
                      dim=self.sys_dim,
                      a = self.a)
    

def vmc_run(Nblocks,Niters,Delta,Walkers_in,Ntarget):
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

                # Metropolis probability for trial move R -> R'
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


def dmc_run(Nblocks,Niters,Walkers,tau,E_T,Ntarget):
    """Diffusion Monte Carlo
    """
    
    max_walkers = 2*Ntarget
    lW=len(Walkers)
    while len(Walkers)<Ntarget:
        Walkers.append(Walkers[max(1,int(lW*random.rand()))].w_copy())

    Eb=zeros((Nblocks,))
    Accept=zeros((Nblocks,))
    AccCount=zeros((Nblocks,))
    
    obs_interval=5
    mass=1

    for i in range(Nblocks):
        EbCount=0
        for j in range(Niters):
            Wsize = len(Walkers)
            Idead = []
            for k in range(Wsize):
                Acc = 0.0
                for np in range(Walkers[k].Ne):
                    R = Walkers[k].Re[np].copy()
                    Psi_R = wfs(Walkers[k])
                    DriftPsi_R = 2*Gradient(Walkers[k],np)/Psi_R*tau/2/mass
                    E_L_R=E_local(Walkers[k])

                    DeltaR=random.randn(Walkers[k].sys_dim)
                    logGf=-0.5*dot(DeltaR,DeltaR)
                    
                    # Move the particle
                    Walkers[k].Re[np]=R+DriftPsi_R+DeltaR*sqrt(tau/mass)
                    
                    # Values for the new position, Rp
                    Psi_Rp = wfs(Walkers[k])
                    DriftPsi_Rp = 2*Gradient(Walkers[k],np)/Psi_Rp*tau/2/mass
                    E_L_Rp = E_local(Walkers[k])
                    
                    DeltaR = R-Walkers[k].Re[np]-DriftPsi_Rp
                    logGb = -dot(DeltaR,DeltaR)/2/tau*mass

                    # Trial move A ( R -> R' )
                    A_RtoRp = min(1, (Psi_Rp/Psi_R)**2*exp(logGb-logGf))

                    # Metropolis
                    if (A_RtoRp > random.rand()):
                        Acc += 1.0/Walkers[k].Ne
                        Accept[i] += 1
                    else:
                        Walkers[k].Re[np]=R
                    
                    AccCount[i] += 1
                
                tau_eff = Acc*tau

                # New branching term
                GB = exp(-(0.5*(E_L_R+E_L_Rp) - E_T)*tau_eff)
                MB = int(floor(GB + random.rand()))
                
                if MB>1:
                    for n in range(MB-1):
                        if (len(Walkers)<max_walkers):
                            Walkers.append(Walkers[k].w_copy())
                elif MB==0:
                    Idead.append(k)
 
            Walkers = DeleteWalkers(Walkers,Idead)

            # Calculate observables every now and then
            if j % obs_interval == 0:
                EL = Observable_E(Walkers)
                Eb[i] += EL
                EbCount += 1
                E_T += 0.01/tau*log(Ntarget/len(Walkers))
                

        Nw = len(Walkers)
        dNw = Ntarget-Nw
        for kk in range(abs(dNw)):
            ind=int(floor(len(Walkers)*random.rand()))
            if (dNw>0):
                Walkers.append(Walkers[ind].w_copy())
            elif dNw<0:
                Walkers = DeleteWalkers(Walkers,[ind])

        
        Eb[i] /= EbCount
        Accept[i] /= AccCount[i]
        #print('Block {0}/{1}'.format(i+1,Nblocks))
        #print('    E   = {0:.5f}'.format(Eb[i]))
        #print('    Acc = {0:.5f}'.format(Accept[i]))


    return Walkers, Eb, Accept

def DeleteWalkers(Walkers,Idead):
    if (len(Idead)>0):
        if (len(Walkers)==len(Idead)):
            Walkers = Walkers[0]
        else:
            Idead.sort(reverse=True)   
            for i in range(len(Idead)):
                del Walkers[Idead[i]]

    return Walkers

def H_1s(r1,r2,a):
    return exp(-a*sqrt(sum((r1-r2)**2)))
     
def wfs(Walker):
    """ Calculate the wave function value by including Jastrow function J(R):
        psi(R) = phi(R)*exp(J(R))
    """

    # H2 approx
    f = H_1s(Walker.Re[0],Walker.Rn[0],Walker.a)+H_1s(Walker.Re[0],Walker.Rn[1],Walker.a)
    f *= (H_1s(Walker.Re[1],Walker.Rn[0],Walker.a)+H_1s(Walker.Re[1],Walker.Rn[1],Walker.a))

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
           J -= Walker.Zn[j]*r/(1.0+100.0*r)
       

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
    """ Returns the kinetic part of local energy
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

    aa=[1.1,1.2,1.3,1.4,1.5]
    es_vmc = []
    vs_vmc = []
    vtes_vmc = []
    # Do the calculations for multiple a
    for a in aa:
        Walkers=[]
        Walkers.append(Walker(Ne=2,
                            Re=[array([0.5,0,0]),array([-0.5,0,0])],
                            spins=[0,1],
                            Nn=2,
                            Rn=[array([-0.70,0,0]),array([0.70,0,0])], # 1.4a_0
                            Zn=[1.0,1.0],
                            dim=3,
                            a = a))
        #print(Walkers[0].Re)
        vmc_only = True
        Ntarget=100
        vmc_time_step = 1.8

        Walkers, Eb, Acc = vmc_run(100,50,vmc_time_step,Walkers,Ntarget)

        E = mean(Eb)
        variance = std(Eb)/sqrt(len(Eb))
        vtoe = abs(variance/E)
        print('\n')
        print('a =                                 ', a)
        print('Energy:                             {:.4f}'.format(E))
        print('Variance:                           {:.4f}'.format(variance))
        print('Absolute variance-to-energy ratio:  {:.4f}'.format(vtoe))
        print('Average acceptance ratio:           {:.4f}'.format(mean(Acc)))

        # Add the results to lists
        es_vmc.append(E)
        vs_vmc.append(variance)
        vtes_vmc.append(vtoe)

    # a = 1.2 provided the best result from VMC get the Walkers again by running
    # vmc first. Sometimes a=1.3 would also be better result, so the best a could
    # potentially be retrieved by calculating the minimum energy from es_vmc (TODO)
    Walkers=[]
    Walkers.append(Walker(Ne=2,
                        Re=[array([0.5,0,0]),array([-0.5,0,0])],
                        spins=[0,1],
                        Nn=2,
                        Rn=[array([-0.70,0,0]),array([0.70,0,0])], # 1.4a_0
                        Zn=[1.0,1.0],
                        dim=3,
                        a = 1.2))
    Walkers, Eb, Accept = vmc_run(100,50,vmc_time_step,Walkers,Ntarget)
    Walkers, Eb_dmc, Accept_dmc = dmc_run(10,10,Walkers,0.05,mean(Eb),Ntarget)

    E = mean(Eb)
    variance = std(Eb)/sqrt(len(Eb))

    E_dmc = mean(Eb_dmc)
    variance_dmc = std(Eb_dmc)/sqrt(len(Eb_dmc))
    vtoe_dmc = abs(variance_dmc/E_dmc)
    
    print('\n')
    print('DMC results')
    print('a =                                 ',a)
    print('Energy:                             {:.4f}'.format(E_dmc))
    print('Variance:                           {:.4f}'.format(variance_dmc))
    print('Absolute variance-to-energy ratio:  {:.4f}'.format(vtoe_dmc))
    print('Average acceptance ratio:           {:.4f}'.format(mean(Accept_dmc)))
    print('')
    print('Total energy comparison of vmc and dmc, a =', a)
    print('E_vmc = {:.4f}'.format(E))
    print('E_dmc = {:.4f}'.format(E_dmc))


    # Plotting total energies with errorbars
    errorbar(aa,es_vmc,vs_vmc,fmt='o-')
    xlabel('a')
    ylabel('E')
    title('Total energies with errorbars, VMC')
    savefig('vmc_energies_errorbar.png')

    # Plotting variance-to-energy ratio
    figure()
    plot(aa,vtes_vmc,'o-')
    xlabel('a')
    ylabel(r'$|\sigma^2 / E|$')
    title('Variance-to-energy ratio, VMC')
    savefig('vmc_variance_to_energy.png')

if __name__=="__main__":
    main()
        
