"""
   FYS-4096 Computational Physics: Exercise 10, Problem 1: time-step extrapolation
   for 2qd

   Temperature has to be kept fixed, so M has to be changed when tau is changed:
        tau = 1/(kTM) <=> T = 1/(k*M*tau). -> M*tau = constant
        To obtain a high temperature we can choose rather small values of tau and M.

        Used values for three points were: tau = 0.3, 0.2, 0.1
                                         ->  M =  50,  75, 150

"""

from numpy import *
from matplotlib.pyplot import *
from scipy.stats import linregress


def main():

    #timesteps
    taus = [0.3, 0.2, 0.1]
    # Energies of 2qd with different time steps
    Es = [2.84686, 3.02936, 3.15938]
    stds = [0.03761, 0.06141, 0.06919]

    # Use linear regression to get intercept and slope 
    linreg = linregress(taus,Es)
    slope = linreg.slope
    intercept = linreg.intercept
    x = array([0,0.5])
    f = slope*x+intercept 

    plot(x,f,'r-',label='Extrapolation of energy')
    plot([0,0.5],[intercept,intercept],'k--',label="Energy at interception: {:.3f} Ha".format(intercept))
    errorbar(taus,Es,yerr=stds,ls='',marker='o')
    xlim(0,0.5)
    ylim(2.6,4)
    title('Extrapolation of energy for two electrom quantum dot')
    xlabel(r'$\tau$')
    ylabel(r'E (Ha)')
    legend()
    savefig('timestep_extrapolation.png')

    show()

if __name__=="__main__":
    main()
        
