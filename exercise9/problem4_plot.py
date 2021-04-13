"""

    FYS-4096 Computational Physics: Exercise 9

    Plots for problem 4
"""

import numpy as np
import matplotlib.pyplot as plt

def main():

    Evmc = -87.848182
    Ejastrow = -90.272104
    Edmc01 = -90.475838
    Edmc005 = -90.477339

    x = np.array([0,1,2,3])
    y = np.array([Evmc,Ejastrow,Edmc01,Edmc005])
    xticks = ['VMC','VMC+Jastrow','DMC, ts=0.01','DMC, ts=0.005',]

    plt.figure(figsize=(8,6))
    plt.plot(x,y,'.',label='Total Energy')
    plt.ylabel('Local energy (Ha)')
    plt.xticks(x, xticks)
    plt.title('Local energy using different techniques')
    plt.savefig('problem4_energy_comparison.png')

    # During Jastrow optimization: 10 different energies
    x = np.linspace(0,9,10)
    y = np.array([-89.822255,-90.213486,-90.278510,-90.271223, \
                  -90.290698,-90.250756,-90.266648,-90.291204, \
                  -90.256553,-90.272104])
    plt.figure()
    plt.plot(x,y,'.')
    plt.xlabel('series (n)')
    plt.xticks(x,['0','1','2','3','4','5','6','7','8','9'])
    plt.ylabel('Local Energy (Ha)')
    plt.title('Local energy during Jastrow factor optimization')
    plt.savefig('problem4_jastrow_optimization.png')

if __name__=='__main__':
    main()