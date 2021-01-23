"""
Cubic hermite splines in 1d, 2d, and 3d

Intentionally unfinished :)

Related to FYS-4096 Computational Physics
exercise 2 assignments.

By Ilkka Kylanpaa on January 2019
"""

from numpy import *
from matplotlib.pyplot import *
from linear_interp import testacc, sqdiff


"""
Add basis functions p1,p2,q1,q2 here
"""
def p1(t):
    return ((1+2*t)*(t-1)**2)

def p2(t):
    return (t**2*(3-2*t))

def q1(t):
    return (t*(t-1)**2)

def q2(t):
    return (t**2*(t-1))


def init_1d_spline(x,f,h):
    # now using complete boundary conditions
    # with forward/backward derivative
    # - natural boundary conditions commented
    a=zeros((len(x),))
    b=zeros((len(x),))
    c=zeros((len(x),))
    d=zeros((len(x),))
    fx=zeros((len(x),))

    # a[0]=1.0 # not needed
    b[0]=1.0

    # natural boundary conditions 
    #c[0]=0.5
    #d[0]=1.5*(f[1]-f[0])/(x[1]-x[0])

    # complete boundary conditions
    c[0]=0.0
    d[0]=(f[1]-f[0])/(x[1]-x[0])
    
    for i in range(1,len(x)-1):
        d[i]=6.0*(h[i]/h[i-1]-h[i-1]/h[i])*f[i]-6.0*h[i]/h[i-1]*f[i-1]+6.0*h[i-1]/h[i]*f[i+1]
        a[i]=2.0*h[i]
        b[i]=4.0*(h[i]+h[i-1])
        c[i]=2.0*h[i-1]        
    #end for

    
    b[-1]=1.0
    #c[-1]=1.0 # not needed

    # natural boundary conditions
    #a[-1]=0.5
    #d[-1]=1.5*(f[-1]-f[-2])/(x[-1]-x[-2])

    # complete boundary conditions
    a[-1]=0.0
    d[-1]=(f[-1]-f[-2])/(x[-1]-x[-2])
    
    # solve tridiagonal eq. A*f=d
    c[0]=c[0]/b[0]
    d[0]=d[0]/b[0]
    for i in range(1,len(x)-1):
        temp=b[i]-c[i-1]*a[i]
        c[i]=c[i]/temp
        d[i]=(d[i]-d[i-1]*a[i])/temp
    #end for
        
    fx[-1]=d[-1]
    for i in range(len(x)-2,-1,-1):
        fx[i]=d[i]-c[i]*fx[i+1]
    #end for
        
    return fx
# end function init_1d_spline

""" 
Add smoothing functions 

def smooth1d(x,f,factor=3):
    ...
    ...
    return ...

def smooth2d(x,y,f,factor=3):
    ...
    ...
    return ... 

def smooth3d(x,y,z,f,factor=3):
    ...
    ...
    ...
    return ...
"""

class spline:

    # Constructor
    def __init__(self,*args,**kwargs):
        self.dims=kwargs['dims']
        if (self.dims==1):
            self.x=kwargs['x']
            self.f=kwargs['f']
            self.hx=diff(self.x)
            self.fx=init_1d_spline(self.x,self.f,self.hx)
        elif (self.dims==2):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.f=kwargs['f']
            self.hx=diff(self.x)
            self.hy=diff(self.y)
            self.fx=zeros(shape(self.f))
            self.fy=zeros(shape(self.f))
            self.fxy=zeros(shape(self.f))

            # Loop through all points
            for i in range(max([len(self.x),len(self.y)])):

                # Find grid points for partial derivatives for the function
                # in terms of x and y
                if (i<len(self.y)):
                    self.fx[:,i]=init_1d_spline(self.x,self.f[:,i],self.hx)
                if (i<len(self.x)):
                    self.fy[i,:]=init_1d_spline(self.y,self.f[i,:],self.hy)
            #end for

            # Find grid point values for fxy, using fy and calculating
            # fxy by calling the function for fx as before
            for i in range(len(self.y)):
                self.fxy[:,i]=init_1d_spline(self.x,self.fy[:,i],self.hx)
            #end for
        elif (self.dims==3):
            self.x=kwargs['x']
            self.y=kwargs['y']
            self.z=kwargs['z']
            self.f=kwargs['f']
            self.hx=diff(self.x)
            self.hy=diff(self.y)
            self.hz=diff(self.z)
            self.fx=zeros(shape(self.f))
            self.fy=zeros(shape(self.f))
            self.fz=zeros(shape(self.f))
            self.fxy=zeros(shape(self.f))
            self.fxz=zeros(shape(self.f))
            self.fyz=zeros(shape(self.f))
            self.fxyz=zeros(shape(self.f))

            # Loop through all points, twice
            for i in range(max([len(self.x),len(self.y),len(self.z)])):
                for j in range(max([len(self.x),len(self.y),len(self.z)])):

                    # Calculating grid points for fx, fy and fz.
                    if (i<len(self.y) and j<len(self.z)):
                        self.fx[:,i,j]=init_1d_spline(self.x,self.f[:,i,j],self.hx)
                    if (i<len(self.x) and j<len(self.z)):
                        self.fy[i,:,j]=init_1d_spline(self.y,self.f[i,:,j],self.hy)
                    if (i<len(self.x) and j<len(self.y)):
                        self.fz[i,j,:]=init_1d_spline(self.z,self.f[i,j,:],self.hz)
            #end for

            # Calculating values for fxy, fxz and fyz
            for i in range(max([len(self.x),len(self.y),len(self.z)])):
                for j in range(max([len(self.x),len(self.y),len(self.z)])):
                    if (i<len(self.y) and j<len(self.z)):
                        self.fxy[:,i,j]=init_1d_spline(self.x,self.fy[:,i,j],self.hx)
                    if (i<len(self.y) and j<len(self.z)):
                        self.fxz[:,i,j]=init_1d_spline(self.x,self.fz[:,i,j],self.hx)
                    if (i<len(self.x) and j<len(self.z)):
                        self.fyz[i,:,j]=init_1d_spline(self.y,self.fz[i,:,j],self.hy)
            #end for

            # Calculating values for fxyz
            for i in range(len(self.y)):
                for j in range(len(self.z)):
                    self.fxyz[:,i,j]=init_1d_spline(self.x,self.fyz[:,i,j],self.hx)
            #end for
        else:
            print('Either dims is missing or specific dims is not available')
        #end if
            
    def eval1d(self,x):
        """
        Evaluates the value of a 1d function using CHS, takes the spline class and the
        space where the function wants to be approximated in, and returns the
        interpolated function
        """
        if isscalar(x):
            x=array([x])
        N=len(self.x)-1
        f=zeros((len(x),))
        ii=0
        for val in x:

            # Finding index
            i=floor(where(self.x<=val)[0][-1]).astype(int)
            if i==N:
                f[ii]=self.f[i]
            else:
                # Calculate the approximated value using the formula for 1D
                t=(val-self.x[i])/self.hx[i]
                f[ii]=self.f[i]*p1(t)+self.f[i+1]*p2(t)+self.hx[i]*(self.fx[i]*q1(t)+self.fx[i+1]*q2(t))
            ii+=1

        return f
    #end eval1d

    def eval2d(self,x,y):
        """
        Evaluates the value of a 2d function using CHS, takes the spline class and the
        space where the function wants to be approximated in, and returns the
        interpolated function
        """
    
        if isscalar(x):
            x=array([x])
        if isscalar(y):
            y=array([y])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        f=zeros((len(x),len(y)))
        A=zeros((4,4))
        ii=0
        for valx in x:
            # Finding the index for x
            i=floor(where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            for valy in y:
                # Finding the index for y
                j=floor(where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1

                # Calculating the tx and ty
                u = (valx-self.x[i])/self.hx[i]
                v = (valy-self.y[j])/self.hy[j]

                # Calculating the left and right side arrays of the 
                # interpolation equation
                pu = array([p1(u),p2(u),self.hx[i]*q1(u),self.hx[i]*q2(u)])
                pv = array([p1(v),p2(v),self.hy[j]*q1(v),self.hy[j]*q2(v)])

                # Calculating the middle array
                A[0,:]=array([self.f[i,j],self.f[i,j+1],self.fy[i,j],self.fy[i,j+1]])
                A[1,:]=array([self.f[i+1,j],self.f[i+1,j+1],self.fy[i+1,j],self.fy[i+1,j+1]])
                A[2,:]=array([self.fx[i,j],self.fx[i,j+1],self.fxy[i,j],self.fxy[i,j+1]])
                A[3,:]=array([self.fx[i+1,j],self.fx[i+1,j+1],self.fxy[i+1,j],self.fxy[i+1,j+1]])           
                
                # Apply dot product twice to obtain the interpolated value
                f[ii,jj]=dot(pu,dot(A,pv))
                jj+=1
            ii+=1
        return f
    #end eval2d

    def eval3d(self,x,y,z):
        """
        Evaluates the value of a 3d function using CHS, takes the spline class and the
        space where the function wants to be approximated in, and returns the
        interpolated function
        """
        if isscalar(x):
            x=array([x])
        if isscalar(y):
            y=array([y])
        if isscalar(z):
            z=array([z])
        Nx=len(self.x)-1
        Ny=len(self.y)-1
        Nz=len(self.z)-1
        f=zeros((len(x),len(y),len(z)))
        A=zeros((4,4))
        B=zeros((4,4))
        ii=0
        for valx in x:
            # Finding index of x
            i=floor(where(self.x<=valx)[0][-1]).astype(int)
            if (i==Nx):
                i-=1
            jj=0
            # Finding index of y
            for valy in y:
                j=floor(where(self.y<=valy)[0][-1]).astype(int)
                if (j==Ny):
                    j-=1
                kk=0
                # Finding index of z
                for valz in z:
                    k=floor(where(self.z<=valz)[0][-1]).astype(int)
                    if (k==Nz):
                        k-=1

                    # Calculating the t values for x,y,z
                    u = (valx-self.x[i])/self.hx[i]
                    v = (valy-self.y[j])/self.hy[j]
                    t = (valz-self.z[k])/self.hz[k]

                    # Calculating the three arrays (functions of u,v,t)
                    pu = array([p1(u),p2(u),self.hx[i]*q1(u),self.hx[i]*q2(u)])
                    pv = array([p1(v),p2(v),self.hy[j]*q1(v),self.hy[j]*q2(v)])
                    pt = array([p1(t),p2(t),self.hz[k]*q1(t),self.hz[k]*q2(t)])

                    # Calculating the A1-4 matrices
                    B[0,:]=array([self.f[i,j,k],self.f[i,j,k+1],self.fz[i,j,k],self.fz[i,j,k+1]])
                    B[1,:]=array([self.f[i+1,j,k],self.f[i+1,j,k+1],self.fz[i+1,j,k],self.fz[i+1,j,k+1]])
                    B[2,:]=array([self.fx[i,j,k],self.fx[i,j,k+1],self.fxz[i,j,k],self.fxz[i,j,k+1]])
                    B[3,:]=array([self.fx[i+1,j,k],self.fx[i+1,j,k+1],self.fxz[i+1,j,k],self.fxz[i+1,j,k+1]])
                    A[:,0]=dot(B,pt)

                    B[0,:]=array([self.f[i,j+1,k],self.f[i,j+1,k+1],self.fz[i,j+1,k],self.fz[i,j+1,k+1]])
                    B[1,:]=array([self.f[i+1,j+1,k],self.f[i+1,j+1,k+1],self.fz[i+1,j+1,k],self.fz[i+1,j+1,k+1]])
                    B[2,:]=array([self.fx[i,j+1,k],self.fx[i,j+1,k+1],self.fxz[i,j+1,k],self.fxz[i,j+1,k+1]])
                    B[3,:]=array([self.fx[i+1,j+1,k],self.fx[i+1,j+1,k+1],self.fxz[i+1,j+1,k],self.fxz[i+1,j+1,k+1]])
                    A[:,1]=dot(B,pt)

                    B[0,:]=array([self.fy[i,j,k],self.fy[i,j,k+1],self.fyz[i,j,k],self.fyz[i,j,k+1]])
                    B[1,:]=array([self.fy[i+1,j,k],self.fy[i+1,j,k+1],self.fyz[i+1,j,k],self.fyz[i+1,j,k+1]])
                    B[2,:]=array([self.fxy[i,j,k],self.fxy[i,j,k+1],self.fxyz[i,j,k],self.fxyz[i,j,k+1]])
                    B[3,:]=array([self.fxy[i+1,j,k],self.fxy[i+1,j,k+1],self.fxyz[i+1,j,k],self.fxyz[i+1,j,k+1]])
                    A[:,2]=dot(B,pt)

                    B[0,:]=array([self.fy[i,j+1,k],self.fy[i,j+1,k+1],self.fyz[i,j+1,k],self.fyz[i,j+1,k+1]])
                    B[1,:]=array([self.fy[i+1,j+1,k],self.fy[i+1,j+1,k+1],self.fyz[i+1,j+1,k],self.fyz[i+1,j+1,k+1]])
                    B[2,:]=array([self.fxy[i,j+1,k],self.fxy[i,j+1,k+1],self.fxyz[i,j+1,k],self.fxyz[i,j+1,k+1]])
                    B[3,:]=array([self.fxy[i+1,j+1,k],self.fxy[i+1,j+1,k+1],self.fxyz[i+1,j+1,k],self.fxyz[i+1,j+1,k+1]])
                    A[:,3]=dot(B,pt)
                
                    # Calculating the interpolated value
                    f[ii,jj,kk]=dot(pu,dot(A,pv))
                    kk+=1
                jj+=1
            ii+=1
        return f
    #end eval3d
#end class spline


    
def main():

    # 1d example
    x=linspace(0.,2.*pi,20)
    y=sin(x)
    spl1d=spline(x=x,f=y,dims=1)
    xx=linspace(0.,2.*pi,100)
    figure()
    # function
    plot(xx,spl1d.eval1d(xx))
    plot(x,y,'o',xx,sin(xx),'r--')
    title('function')

    # Test points and indeces
    xind = np.array([13,16,23,42,73,97])
    xpts = xx[xind]
    fvals = np.sin(xpts)

    # get interpolated values
    func = spl1d.eval1d(xx)
    interpvals = func[xind]
    err1 = testacc(fvals,interpvals, "1d")

    
    # 2d example
    fig=figure()
    ax=fig.add_subplot(121)
    x=linspace(0.0,3.0,11)
    y=linspace(0.0,3.0,11)
    X,Y = meshgrid(x,y)
    Z = (X+Y)*exp(-1.0*(X*X+Y*Y))
    ax.pcolor(X,Y,Z)
    ax.set_title('original')

    spl2d=spline(x=x,y=y,f=Z,dims=2)
    #figure()
    ax2=fig.add_subplot(122)
    x=linspace(0.0,3.0,51)
    y=linspace(0.0,3.0,51)
    X,Y = meshgrid(x,y)
    Z = spl2d.eval2d(x,y)
    ax2.pcolor(X,Y,Z)
    ax2.set_title('interpolated')

    # 2d testing
    # Test points and indeces
    xind = np.array([13,16,23,32,43,23,5])
    yind = np.array([35,13,12,49,22,6,46])
    xpts = x[xind]
    ypts = y[yind]
    fvals = xpts*(np.exp(-1.0*(xpts*xpts+ypts*ypts)))

    # get interpolated values
    interpvals = Z[xind,yind]
    err2 = testacc(fvals,interpvals, "2d")

    # 3d example
    x=linspace(0.0,3.0,10)
    y=linspace(0.0,3.0,10)
    z=linspace(0.0,3.0,10)
    X,Y,Z = meshgrid(x,y,z)
    F = (X+Y+Z)*exp(-1.0*(X*X+Y*Y+Z*Z))
    X,Y= meshgrid(x,y)
    fig3d=figure()
    ax=fig3d.add_subplot(121)
    ax.pcolor(X,Y,F[...,int(len(z)/2)])
    ax.set_title('original')

    spl3d=spline(x=x,y=y,z=z,f=F,dims=3)  
    x=linspace(0.0,3.0,50)
    y=linspace(0.0,3.0,50)
    z=linspace(0.0,3.0,50)
    X,Y= meshgrid(x,y)
    ax2=fig3d.add_subplot(122)
    F=spl3d.eval3d(x,y,z)
    ax2.pcolor(X,Y,F[...,int(len(z)/2)])
    ax2.set_title('interpolated')

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
    err3 = testacc(fvals,interpvals, "3d")

    # Error plot
    dims = ["1d","2d","3d"]
    errs = [err1,err2,err3]
    errfig = figure()
    axerr = errfig.add_subplot(111)
    axerr.bar(dims,errs)
    axerr.set_ylabel("Average squared error")
    axerr.set_title("Average squared error for 1d, 2d and 3d cases")

    show()
#end main
    
if __name__=="__main__":
    main()
