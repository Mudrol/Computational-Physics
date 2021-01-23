"""
Computational Physics Exercise 2, Problem 3: Root search
This program includes the routines used for finding indices
needed in interpolation.

By Matias Hiillos, 2021
"""

import numpy as np

def find_indices_bsearch(grid, low, high, rval):
    """
    Function finds the indices of the grid that the value is inside of by using
    binary search. The return value is the index where the value is lower (i),
    and the next index is simply i+1.

    Prerequisites: rval is inside the grid

    Parameters:
    grid, array to find the value indeces in
    low, lowest index of array
    high, highest index of array
    rval, the value to find the indeces for
    """
    

    if high >= low:

        mid = (high+low) // 2

        # If the same value, return it
        if grid[mid] == rval:
            return mid
        
        # If the current index is higher, check the index below
        elif grid[mid] > rval:

            # If the lower index is below the value, we have found the place
            # in grid
            if grid[mid-1] < rval:
                return mid-1
            
            # Otherwise, call the function recursively for the left part of the
            # array
            else:
                return find_indices_bsearch(grid, low, mid - 1, rval)
        else:

            # If midpoint is lower and one higher index is higher, return the index
            if grid[mid+1] > rval:
                return mid

            # Otherwise, call the function recursively for the right part of
            # the array    
            else:
                return find_indices_bsearch(grid, mid + 1, high, rval)
    
    else:
        # rval outside of the grid
        return -1



def main():
    
    # Linear grid
    grid = np.linspace(0,2*np.pi,10)

    # Random number inside the grid [0,10)
    rval = float(2*np.pi*np.random.random(1,))
    print(rval)

    i = find_indices_bsearch(grid,0,len(grid)-1,rval)
    print("Random value:", rval)
    print("Indeces for random value are " + str(i) + " and " + str(i+1) + "." )
    print("Values for the grid at these indeces are", grid[i], "and", grid[i+1])

    # b)
    rmax = 100
    r_0 = 10 ** -5
    dim = 100
    r = np.zeros(dim)
    r[0] = 0.
    h = np.log(rmax/r_0+1)/(dim-1)
    for i in range(1,dim):
        r[i] = r_0*(np.exp(i*h)-1)
    
    # Random number inside the grid
    rval2 = float(r[len(r)-1]*np.random.random(1,))
    print(rval2)
    ii = find_indices_bsearch(r,0,len(r)-1,rval2)
    print("Random value:", rval2)
    print("Indeces for random value are " + str(ii) + " and " + str(ii+1) + "." )
    print("Values for the grid at these indeces are", r[ii], "and", r[ii+1])



if __name__ == "__main__":
    main()