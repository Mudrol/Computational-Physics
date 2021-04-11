""" 
    FYS-4096 Computational Physics Exercise 10, Problem 3
    
    Plotting 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def main():
    filenames = ['S0','S3']
    conv_cut = 5 # Index for equilibration cutoff
    for i in range(2):
        density = np.loadtxt('ag_density_'+filenames[i]+'.dat') # x y z

        X,Y,Z = [density[:,0],density[:,1],density[:,2]]

        Etot = np.loadtxt('E_tot_'+filenames[i]+'.dat') # 1d

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_trisurf(X,Y,Z,cmap='viridis',antialiased=True,linewidth=0.2)
        ax.set_title('Electron density for '+filenames[i])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.savefig('density_plot_'+filenames[i]+'.png')

        plt.figure()
        plt.plot(Etot)
        plt.plot([conv_cut,conv_cut],plt.gca().get_ylim(),'k--') 
        plt.xlabel('m')
        plt.ylabel(r'$\langle E \rangle$ (Ha)')
        mean = np.mean(Etot[conv_cut::])
        std = np.std(Etot[conv_cut::])
        estr = r'Energy mean = {:.3f} Ha'.format(mean)
        stdstr = r'$\sigma$ = {:.3f} Ha'.format(std)
        etext = mpatches.Patch(color='white',label=estr)
        stdtext = mpatches.Patch(color='white',label=stdstr)
        plt.legend(handles=[etext,stdtext])


        plt.savefig('Etot_plot_'+filenames[i]+'.png')
    Etot_S0 = np.loadtxt('E_tot_S0.dat')
    Etot_S3 = np.loadtxt('E_tot_S3.dat')
    S0_mean = np.mean(Etot_S0[conv_cut::])
    S3_mean = np.mean(Etot_S3[conv_cut::])
    S0_std = np.std(Etot_S0[conv_cut::])
    S3_std = np.std(Etot_S3[conv_cut::])


    x_pos = np.arange(len(filenames))
    avgs = [S0_mean, S3_mean]
    stds = [S0_std, S3_std]

    fig, ax = plt.subplots()
    ax.bar(x_pos, avgs, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(r'$E_{tot}$ (Ha)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(filenames)
    ax.set_title('Total energy for states S=0, S=3')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('energy_plot_with_errbar.png')

if __name__=="__main__":
    main()