"""

    FYS-4096 Computational Physics: Exercise 8

    Plotting of the potential energy for O2 molecule using data from
    CSC
"""

import numpy as np
import spline_class
import matplotlib.pyplot as plt


def main():

    # experimental equilibrium value
    d = 1.2074 # Angstrom

    # Points around the value used for calculating the energies
    ds = np.linspace(d-0.15,d+0.15,5)

    # Energies gotten from calculations ( unit: Ry )
    ens = [-63.60581888, -63.68028866, -63.70049255, -63.68788723, -63.65654726]

    # 1d spline
    spl1d = spline_class.spline(x=ds,f=ens,dims=1)
    x = np.linspace(ds[0],ds[-1],100)


    plt.figure()

    # Plot the spline
    plt.plot(x,spl1d.eval1d(x),'r--',label='spline')

    # Numpy's polyfit: parabolic fit. Fit around the minimum points, making the fit
    # more accurate
    fitted_ens = np.polyfit(ds[1:-1],ens[1:-1],2)
    p = np.poly1d(fitted_ens)
    plt.plot(x,p(x),'g',label='2nd order polynomial')

    # Plot the datapoints 
    plt.plot(ds,ens,'.',markersize='12')

    plt.legend()
    plt.xlabel('R (Å)')
    plt.ylabel('E (Ry)')
    plt.title(r'Potential energy surface of $O_2$ near $d_{eq}=1.2074  Å$')
    
    plt.savefig('problem3_plot.png')
if __name__=='__main__':
    main()