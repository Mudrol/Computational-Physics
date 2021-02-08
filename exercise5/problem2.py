"""
    FYS-4096 Computational Physics: Exercise 5
    Problem 2: Trajectory of a charged particle influenced by both electric and magnetic field

    This program calculates the trajectory of a particle in a constant
    magnetic and electric field. odeint -function from scipy.integrate is used
    to solve the differential equations

    Made by: Matias Hiillos
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def eom_lf(Q, t, E, B, qem, qbm):
    """
    Equation of motion for the particle under a constant electric and magnetic field.

    Arguments:
    Q: Array with values for the position and velocity of the particle
       (x,y,z,vx,vy,vz)
    t: time
    E = Electric field component
    B = Magnetic field component
    qem = qE/m
    qbm = qB/m

    Return: Array with dr/dt and dv/dt values
    """

    v = Q[3:]

    drdt = v
    dvdt = qem * E + qbm * np.cross(v,B)

    return np.concatenate((drdt, dvdt))


def main():
    """
    Trajectory of a particle in a constant magnetic and electric field. 
    ODE solved with odeint.
    """

    # Constants and fields in arbitrary units
    B = np.array([0,1,0])
    E = np.array([1,0,0])
    qem = 0.05
    qbm = 4.0

    # Initial values
    v_0 = np.ones(3)*0.1
    r_0 = np.zeros(3)
    Q_0 = np.concatenate((r_0,v_0))

    t = np.linspace(0,5,100)
    Q = odeint(eom_lf, Q_0, t, args=(E,B,qem,qbm))

    # Calculating the coordinates and the velocity vector at time t = 5
    r_5 = np.array([Q[len(t)-1,0],Q[len(t)-1,1],Q[len(t)-1,2]])
    v_5 = np.array([Q[len(t)-1,3],Q[len(t)-1,4],Q[len(t)-1,5]])

    print("Coordinates at time t=5:", r_5)
    print("Velocity vector at time t=5:", v_5)
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Q[:,0],Q[:,1],Q[:,2])
    plt.title("Trajectory of a particle for $t \in [0,5]$")

    plt.show()


if __name__=="__main__":
    main()