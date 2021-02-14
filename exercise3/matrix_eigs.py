"""
FYS-4096 Computational physics 

1. Add code to function 'largest_eig'
- use the power method to obtain 
  the largest eigenvalue and the 
  corresponding eigenvector of the
  provided matrix

2. Compare the results with scipy's eigs
- this is provided, but you should use
  that to validating your power method
  implementation

Hint: 
  dot(A,x), A.dot(x), A @ x could be helpful for 
  performing matrix operations

"""


from numpy import *
from matplotlib.pyplot import *
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps


def largest_eig(A,tol=1e-12):
    """
    - Simple power method code needed in this function.
    - Also good commenting.
    - You may modify the output / return values if you like
    """

    # Initialize eigenvector, eigenvalue and the precision
    # Randomizing the eigenvector gives much more accurate estimation
    eig_vector = np.random.rand(A.shape[0])
    eig_value = 0.
    prec = 2*tol

    # Using the power method until at precision
    while prec > tol:
      vk = A.dot(eig_vector) # A*x

      # Normalizing the vector 
      x_norm = np.linalg.norm(vk)
      x_1 = vk/x_norm

      # Updating the eigenvalue, A*A*x etc.
      eig_value = x_1.dot(A.dot(x_1))

      # Updating the precision for tolerance check
      prec = np.linalg.norm(vk-eig_value*eig_vector)

      # Updating eigenvector
      eig_vector = x_1

    return eig_value, eig_vector


def main():
    grid = linspace(-5,5,100)
    grid_size = grid.shape[0]
    dx = grid[1]-grid[0]
    dx2 = dx*dx
    
    # make test matrix
    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size) - 1.0/(abs(grid)+0.8),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])

    # use scipy to calculate the largest eigenvalue
    # and corresponding vector
    eigs, evecs = sla.eigsh(H0, k=1, which='LA')

    # use your power method to calculate the same
    l,vec=largest_eig(H0)
    
    # see how they compare
    print('largest_eig estimate: ', l)
    print('scipy eigsh estimate: ', eigs)
    
    # eigsh eigen vector
    psi0=evecs[:,0]
    norm_const=simps(abs(psi0)**2,x=grid)
    psi0=psi0/norm_const
    
    # largest_eig eigen vector 
    # At the moment works without [:,0] as the current power method function
    # only returns the eigenvector corresponding to the largest eigenvalue 
    psi0_=vec
    norm_const=simps(abs(psi0_)**2,x=grid)
    psi0_=psi0_/norm_const
    
    plot(grid,abs(psi0)**2,label='scipy eig. vector squared')
    plot(grid,abs(psi0_)**2,'r--',label='largest_eig vector squared')
    legend(loc=0)
    show()


if __name__=="__main__":
    main()
