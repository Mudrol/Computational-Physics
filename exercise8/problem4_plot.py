"""

    FYS-4096 Computational Physics: Exercise 8

    Plotting of the potential energy for O2 molecule using data from
    CSC
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():

    # k-grid sizes
    kgrids = [1,2,3,4]

    # energy cut-off values
    ecuts = [100,150,200,250]

    # Energies for different cut-off values
    etots_c = [-22.51487991,-22.52203164,-22.52257768,-22.52262441]

    # Energies for different k-grids
    etots_k = [-20.54485326,-22.52257768,-22.71847590,-22.75252416]
    
    plt.figure(figsize=(8,6))

    plt.plot(kgrids,etots_k,'.',label='Total Energy')
    plt.xlabel('k-grid size (i,i,i)')
    plt.ylabel('Total energy (Ry)')
    plt.xticks(kgrids, ['1','2','3','4'])
    plt.title('Convergence of total energy respect to k-grid size')

    plt.savefig('problem4_kgrid_plot.png')

    plt.figure(figsize=(8,6))

    plt.plot(ecuts,etots_c,'.')
    plt.xlabel('Energy cut-off (Ry)')
    plt.ylabel('Total energy (Ry)')
    plt.title('Convergence of total energy respect to energy-cutoff values')
    plt.savefig('problem4_ecut_plot.png')

if __name__=='__main__':
    main()