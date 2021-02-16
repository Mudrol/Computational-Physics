"""
    FYS-4096 Computational Physics: Exercise 6
    Problem 2: 1D FEM for Poisson equation

    Solves the 1d Poisson equation in three ways:

    #1:    Finite Element Method without integration
    #2:    Finite Element Method with integration
    #3:    Finite Difference Method

    Made by: Matias Hiillos
"""

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

eps_0 = 8.854*1e-11

def FEM1D(N):
    """
    Solves the 1D Poisson equation without integration

    N: Amount of gridpoints
    """
    x = np.linspace(0,1,N)
    h = x[1] - x[0]
    b = np.zeros(N)

    # A matrix, diagonal and off-diagonal values
    # according to formula given in the exercise
    diags = 2/h * np.ones(N)
    offdiags = -1/h * np.ones(N-1)
    A = np.diag(diags) + np.diag(offdiags,1) + np.diag(offdiags,-1)

    # Apply periodic boundary conditions
    # Works because sin(x*pi) = 0 when x=0,1
    A[0][1] = A[-1][-2] = 0

    # b vector
    for i in range(1, N-1):
        b[i] = np.pi/h * (x[i-1]+x[i+1]-2*x[i])*np.cos(np.pi*x[i]) + \
               1/h * (2*np.sin(np.pi*x[i])-np.sin(np.pi*x[i-1])-np.sin(np.pi*x[i+1]))
    
    # Solve the equation and return the result
    return np.linalg.solve(A,b)

def FEM1D_int(N):
    """
    Solves the 1D Poisson equation with integration
    """
    x = np.linspace(0,1,N)
    h = x[1] - x[0]
    A = np.zeros((N,N))
    b = np.zeros(N)

    # More points for the hat functions u
    # below 6*N doesn't give accurate values for example N = 7, but for
    # current N (15) it works with 4*N. Increasing this parameter both increases
    # accuracy and computation time.
    M = 4*N
    x_u = np.linspace(0,1,M)
    h_u = x_u[1] - x_u[0]

    # Calculate the values of A matrix by integrating over the derivatives
    # of the hat functions:
    # ( A_ij = integral( u'_j*u'_i ) )
    for i in range(N):
        ui = hat_func(x_u, i, x, h)
        for j in range(N):
            uj = hat_func(x_u, j, x, h)
            u_ddots = np.gradient(ui,h_u) * np.gradient(uj,h_u)
            A[i,j] = simps(u_ddots,x_u)

    # Calculating the b vector by integration:
    # ( b_i = 1/eps_0 * integral( rho*u_i ) )
    for i in range(1,N-1):
        ui = hat_func(x_u, i, x, h)
        b[i] = simps(rho(x_u)*ui,x_u)/eps_0

    # Solve the equation and return the result
    return np.linalg.solve(A,b)

def FDM1D(N):
    """
    Solves the problem using Finite Difference Method
    """
    x = np.linspace(0,1,N)
    h = x[1] - x[0]

    # Second centered difference: phi'' = 1/h^2 * (phi[i+1] - 2phi[i] + phi[i-1])
    # Create the matrix A
    diags = -2*np.ones(N)
    offdiags = np.ones(N-1)

    A = (np.diag(diags) + np.diag(offdiags, 1) + np.diag(offdiags, -1))/(h**2)

    # Apply periodic boundary conditions
    # Works because sin(x*pi) = 0 when x=0,1
    A[0][1] = A[-1][-2] = 0

    # Create the b vector ( b = -rho/eps_0)
    # Boundary terms included: sin(x*pi) = 0 when x = 0 or 1
    b = -1*rho(x)/eps_0
    print(b)

    # Solve the linear system
    return np.linalg.solve(A,b)


def hat_func(x_u, i, x, h):
    """
    Calculates the hat function u_i values, used for determining the values of
    matrix A and vector b.

    parameters:
    x_u: Values to evaluate u_i in
    i: index to the value to evaluate inside
    x: grid
    h: grid separation

    returns the values of u_i
    """
    u = np.zeros(len(x_u))

    # Make the evaluation at every point of u_i
    for j in range(len(x_u)-1):
        val = x_u[j]

        # Update the value according to the formula
        if x[i-1] < val < x[i]:
            u[j] = (val - x[i-1])/h

        elif x[i] < val < x[i+1]:
            u[j] = (x[i+1] - val)/h

        else:
            u[j] = 0

    return u

def rho(x):
    """
    Function given in the exercise
    """
    return eps_0*np.pi**2*np.sin(np.pi*x)

def phi_sol(x):
    """
    Analytical solution
    """
    return np.sin(np.pi*x)

def main():

    N = 15
    phi = FEM1D(N)
    phi_int = FEM1D_int(N)
    phi_fdm = FDM1D(N)

    # Coordinates for plotting and calculating error
    x = np.linspace(0,1,N)      # Numerical methods
    xx = np.linspace(0,1,100)   # Analytical values

    # Calculate the maximum differences compared to analytical values
    err_fem1d = np.max(np.abs(phi-phi_sol(x)))
    err_fem1d_int = np.max(np.abs(phi_int-phi_sol(x)))
    err_fdm1d = np.max(np.abs(phi_fdm-phi_sol(x)))

    print("Maximum absolute error of FEM without integration: ", err_fem1d)
    print("Maximum absolute error of FEM with integration: ", err_fem1d_int)
    print("Maximum absolute error of FDM method: ", err_fdm1d)

    plt.figure()
    plt.plot(x,phi,label="FEM")
    plt.plot(xx,phi_sol(xx),label="analytic")
    plt.plot(x,phi_int,label="FEM with integration")
    plt.plot(x,phi_fdm,label="FDM")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()