"""
Linear interpolation in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2 assignments.

By Ilkka Kylanpaa on January 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
Add basis functions l1 and l2 here
"""
def l1(t):
    return 1-t

def l2(t):
    return t

class linear_interp:

    # Constructor
    def __init__(self,*args,**kwargs):
        self.dims=kwargs['dims']
        if (self.dims==1):
            self.x=kwargs['x']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
        elif (self.dims==2):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
        elif (self.dims==3):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.z=kwargs['z']
            self.f=kwargs['f']
            self.hx=np.diff(self.x)
            self.hy=np.diff(self.y)
            self.hz=np.diff(self.z)
        else:
            print('Either dims is missing or specific dims is not available')
      
    def eval1d(self,x):
        """
        1D interpolation for function. Takes linear_interp class with one
        dimension and 1D array as the section where to do the interpolation.
        Returns the interpolated function as a 1d array.
        """

        # Force x to be an array
        if np.isscalar(x):
            x=np.array([x])
        N=len(self.x)-1
        f=np.zeros((len(x),))
        ii=0

        for val in x:

            # Finding the index
            i=np.floor(np.where(self.x<=val)[0][-1]).astype(int)

            # If the last index, add it to the function to be returned
            if i==N:
                f[ii]=self.f[i]
            else:

                # Calculate the approximated function value with the formula
                t=(val-self.x[i])/self.hx[i]
                f[ii]=self.f[i]*l1(t)+self.f[i+1]*l2(t)
            ii+=1
        return f

    def eval2d(self,x,y):
        """
        2d interpolation. Takes linear_interp class with two
        dimensions and two 1d arrays as the area where to do the interpolation.
        Returns the interpolated function as a 2d array.
        """

        # Force x and y to be an array
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        f=np.zeros((len(x),len(y)))
        A=np.zeros((2,2))
        ii=0
        for valx in x:
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)

            # If the last index, substract one to avoid out of range during
            # calculation
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1

                # Calculate the approximated function value with the 2d formula
                tx = (valx-self.x[i])/self.hx[i]
                ty = (valy-self.y[j])/self.hy[j]
                ptx = np.array([l1(tx),l2(tx)])
                pty = np.array([l1(ty),l2(ty)])
                A[0,:]=np.array([self.f[i,j],self.f[i,j+1]])
                A[1,:]=np.array([self.f[i+1,j],self.f[i+1,j+1]])
                f[ii,jj]=np.dot(ptx,np.dot(A,pty))
                jj+=1
            ii+=1
        return f
    #end eval2d

    def eval3d(self,x,y,z):
        """
        3d interpolation. Takes linear_interp class with three
        dimensions and three 1d arrays as the area where to do the interpolation.
        Returns the interpolated function as a 3d array.
        """

        # Force x, y and z to be an array
        if np.isscalar(x):
            x=np.array([x])
        if np.isscalar(y):
            y=np.array([y])
        if np.isscalar(z):
            z=np.array([z])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        Nz=len(self.z)-1
        f=np.zeros((len(x),len(y),len(z)))
        A=np.zeros((2,2))
        B=np.zeros((2,2))
        ii=0
        for valx in x:

            # If the last index, substract one to avoid out of range during
            # calculation
            i=np.floor(np.where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                j=np.floor(np.where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                kk=0
                for valz in z:
                    k=np.floor(np.where(self.z<=valz)[0][-1]).astype(int)
                    if (k==Nz):
                        k-=1

                    # Calculate the approximated function 
                    # value with the 3d formula
                    tx = (valx-self.x[i])/self.hx[i]
                    ty = (valy-self.y[j])/self.hy[j]
                    tz = (valz-self.z[k])/self.hz[k]
                    ptx = np.array([l1(tx),l2(tx)])
                    pty = np.array([l1(ty),l2(ty)])
                    ptz = np.array([l1(tz),l2(tz)])
                    B[0,:]=np.array([self.f[i,j,k],self.f[i,j,k+1]])
                    B[1,:]=np.array([self.f[i+1,j,k],self.f[i+1,j,k+1]])
                    A[:,0]=np.dot(B,ptz)
                    B[0,:]=np.array([self.f[i,j+1,k],self.f[i,j+1,k+1]])
                    B[1,:]=np.array([self.f[i+1,j+1,k],self.f[i+1,j+1,k+1]])
                    A[:,1]=np.dot(B,ptz)
                    f[ii,jj,kk]=np.dot(ptx,np.dot(A,pty))
                    kk+=1
                jj+=1
            ii+=1
        return f
    #end eval3d
# end class linear interp


def sqdiff(s, f):
    """
    Calculates the square difference between the analytical and the interpolated 
    function value and returns it.
    """
    return ((f-s)**2)


#xpts,fvals,interpvals
def testacc(f, s, dim):
    """
    Tests the accuracy of the interpolation with the average of squared difference.
    Takes the analytical values f and interpolated values s as arrays, and also
    the dimension we are testing as a string.
    """
    totdiff = 0.

    # Calculate squared differences for all test points
    for i in range(len(f)):
        totdiff += sqdiff(s[i],f[i])
    sqdavg = totdiff/len(f)
    print("Average of squared difference in", dim, "with", len(f), \
          "test points:", "{:0.4f}".format(sqdavg))


def main():

    fig1d = plt.figure()
    ax1d = fig1d.add_subplot(111)

    # 1d example
    x=np.linspace(0.,2.*np.pi,10)
    y=np.sin(x)
    lin1d=linear_interp(x=x,f=y,dims=1)
    xx=np.linspace(0.,2.*np.pi,100)
    ax1d.plot(xx,lin1d.eval1d(xx))
    ax1d.plot(x,y,'o',xx,np.sin(xx),'r--')
    ax1d.set_title('function')

    # Test points and indeces
    xind = np.array([13,16,23,63,73,97])
    xpts = xx[xind]
    fvals = np.sin(xpts)

    # get interpolated values
    func = lin1d.eval1d(xx)
    interpvals = func[xind]
    testacc(fvals,interpvals, "1d")


    # 2d example
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(221, projection='3d')
    ax2d2 = fig2d.add_subplot(222, projection='3d')
    ax2d3 = fig2d.add_subplot(223)
    ax2d4 = fig2d.add_subplot(224)

    x=np.linspace(-2.0,2.0,11)
    y=np.linspace(-2.0,2.0,11)
    X,Y = np.meshgrid(x,y)
    Z = X*np.exp(-1.0*(X*X+Y*Y))
    ax2d.plot_wireframe(X,Y,Z)
    ax2d3.pcolor(X,Y,Z)
    #ax2d3.contourf(X,Y,Z)

    lin2d=linear_interp(x=x,y=y,f=Z,dims=2)
    x=np.linspace(-2.0,2.0,51)
    y=np.linspace(-2.0,2.0,51)
    X,Y = np.meshgrid(x,y)
    Z = lin2d.eval2d(x,y)
     
    ax2d2.plot_wireframe(X,Y,Z)
    ax2d4.pcolor(X,Y,Z)

    # 2d testing
    # Test points and indeces
    xind = np.array([13,16,23,32,43,23, 5])
    yind = np.array([35,13,12,49,22,6, 46])
    xpts = x[xind]
    ypts = y[yind]
    fvals = xpts*(np.exp(-1.0*(xpts*xpts+ypts*ypts)))

    # get interpolated values
    interpvals = Z[xind,yind]
    testacc(fvals,interpvals, "2d")

    
    # 3d example
    x=np.linspace(0.0,3.0,10)
    y=np.linspace(0.0,3.0,10)
    z=np.linspace(0.0,3.0,10)
    X,Y,Z = np.meshgrid(x,y,z)
    F = (X+Y+Z)*np.exp(-1.0*(X*X+Y*Y+Z*Z))
    X,Y= np.meshgrid(x,y)
    fig3d=plt.figure()
    ax=fig3d.add_subplot(121)
    ax.pcolor(X,Y,F[...,int(len(z)/2)])
    lin3d=linear_interp(x=x,y=y,z=z,f=F,dims=3)
    
    x=np.linspace(0.0,3.0,50)
    y=np.linspace(0.0,3.0,50)
    z=np.linspace(0.0,3.0,50)
    X,Y= np.meshgrid(x,y)
    F=lin3d.eval3d(x,y,z)
    ax2=fig3d.add_subplot(122)
    ax2.pcolor(X,Y,F[...,int(len(z)/2)])

    #3d testing
    # Test points and indeces
    xind = np.array([13,37,24,12,42,7,3])
    yind = np.array([1,42,23,35,12,32,14])
    zind = np.array([43,17,34,12,7,46,12])
    xpts = x[xind]
    ypts = y[yind]
    zpts = z[zind]
    fvals = (xpts+ypts+zpts)*np.exp(-1.0*(xpts*xpts+ypts*ypts+zpts*zpts))

    # get interpolated values
    interpvals = F[xind,yind,zind]
    testacc(fvals,interpvals, "3d")

    # TODO: PLOT ERRORS!
    #plterr = plt.figure()


    plt.show()


#end main
    
if __name__=="__main__":
    main()
