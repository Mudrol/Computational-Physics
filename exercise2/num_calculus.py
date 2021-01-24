"""
This module contains functions for numerical calculus:
- first and second derivatives
- 1D integrals: Riemann, trapezoid, Simpson, 
  and Monte Carlo with uniform random numbers
- N-dimensional numerical gradient
"""

import numpy as np

def first_derivative(function, x, dx ):
    """ 
    This calculates the first derivative with
    symmetric two point formula, which has O(h^2)
    accuracy. See, e.g., FYS-4096 lecture notes.
    """
    return (function(x+dx)-function(x-dx))/2/dx

def second_derivative(function, x, dx):
    """ 
    This calculates the second derivative with
    O(h^2) accuracy. See, e.g., FYS-4096 lecture 
    notes.
    """
    return (function(x+dx)+function(x-dx)-2.*function(x))/dx**2

def riemann_sum(function,x):
    """ 
    Left Riemann sum for uniform grid. 
    See, e.g., FYS-4096 lecture notes.
    """
    dx=x[1]-x[0]
    f=function(x)
    return np.sum(f[0:-1])*dx

def trapezoid(function,x):
    """ 
    Trapezoid for uniform grid. 
    See, e.g., FYS-4096 lecture notes.
    """
    dx=x[1]-x[0]
    f=function(x)
    return (f[0]/2+np.sum(f[1:-1])+f[-1]/2)*dx

def simpson_integration(function,x):
    """ 
    Simpson rule for uniform grid 
    See, e.g., FYS-4096 lecture notes.
    """
    f=function(x)
    N = len(x)-1
    dx=x[1]-x[0]
    s0=s1=s2=0.
    for i in range(1,N,2):
        s0+=f[i]
        s1+=f[i-1]
        s2+=f[i+1]
    s=(s1+4.*s0+s2)/3
    if (N+1)%2 == 0:
        return dx*(s+(5.*f[N]+8.*f[N-1]-f[N-2])/12)
    else:
        return dx*s

def simpson_nonuniform(function,x):
    """ 
    Simpson rule for nonuniform grid 
    See, e.g., FYS-4096 lecture notes.
    """
    f = function(x)
    N = len(x)-1
    h = np.diff(x)
    s=0.
    for i in range(1,N,2):
        hph=h[i]+h[i-1]
        s+=f[i]*(h[i]**3+h[i-1]**3+3.*h[i]*h[i-1]*hph)/6/h[i]/h[i-1]
        s+=f[i-1]*(2.*h[i-1]**3-h[i]**3+3.*h[i]*h[i-1]**2)/6/h[i-1]/hph
        s+=f[i+1]*(2.*h[i]**3-h[i-1]**3+3.*h[i-1]*h[i]**2)/6/h[i]/hph
    if (N+1)%2 == 0:
        s+=f[N]*(2.*h[N-1]**2+3.*h[N-2]*h[N-1])/6/(h[N-2]+h[N-1])
        s+=f[N-1]*(h[N-1]**2+3.*h[N-1]*h[N-2])/6/h[N-2]
        s-=f[N-2]*h[N-1]**3/6/h[N-2]/(h[N-2]+h[N-1])
    return s

def monte_carlo_integration(fun,xmin,xmax,blocks,iters):
    """ 
    1D Monte Carlo integration with uniform random numbers
    in range [xmin,xmax]. As output one gets the value of 
    the integral and one sigma statistical error estimate,
    that is, ~68% reliability. Two sigma and three sigma
    estimates are with ~95% and ~99.7% reliability, 
    respectively. See, e.g., FYS-4096 lecture notes. 
    """
    block_values=np.zeros((blocks,))
    L=xmax-xmin
    for block in range(blocks):
        for i in range(iters):
            x = xmin+np.random.rand()*L
            block_values[block]+=fun(x)
        block_values[block]/=iters
    I = L*np.mean(block_values)
    dI = L*np.std(block_values)/np.sqrt(blocks)
    return I, dI 

def  numerical_gradient(f,r,dx):
    """
    N-dimensional numerical gradient, first derivatives calculated with
    symmetric two-point formula, which has accuracy of O(h^2). Returns the
    gradient as a array of length N.
    """

    # Check for 1D case
    if np.isscalar(r): 
        return first_derivative(f,r,dx)
    else:
        N = len(r)
        grad = np.zeros(N)
        for i in range(N):
            grad[i] = (f(r+dx)[i]-f(r-dx)[i])/2/dx[i]
        return grad

""" Test routines for unit testing """
def test_first_derivative(tolerance=1.0e-3):
    """ Test routine for first derivative of f"""
    x = 0.8
    dx = 0.01
    df_estimate = first_derivative(test_fun,x,dx)
    df_exact = test_fun_der(x)
    err = np.abs(df_estimate-df_exact)
    working = False
    if (err<tolerance):
        print('First derivative is OK')
        working = True
    else:
        print('First derivative is NOT ok!!')
    return working

def test_second_derivative(tolerance=1.0e-3):
    """ Test routine for first derivative of f"""
    x = 0.8
    dx = 0.01
    df_estimate = second_derivative(test_fun,x,dx)
    df_exact = test_fun_der2(x)
    err = np.abs(df_estimate-df_exact)
    working = False
    if (err<tolerance):
        print('Second derivative is OK')
        working = True
    else:
        print('Second derivative is NOT ok!!')
    return working

def test_riemann_sum(tolerance=1.0e-2):
    """ Test routine for Riemann integration"""
    a = 0
    b = np.pi/2
    x = np.linspace(a,b,100)
    int_estimate = riemann_sum(test_fun2,x)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_estimate-int_exact)
    working = False
    if (err<tolerance):
        print('Riemann integration is OK')
        working = True
    else:
        print('Riemann integration is NOT ok!!')
    return working

def test_trapezoid(tolerance=1.0e-4):
    """ Test routine for trapezoid integration"""
    a = 0
    b = np.pi/2
    x = np.linspace(a,b,100)
    int_estimate = trapezoid(test_fun2,x)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_estimate-int_exact)
    working = False
    if (err<tolerance):
        print('Trapezoid integration is OK')
        working = True
    else:
        print('Trapezoid integration is NOT ok!!')
    return working

def test_simpson_integration(tolerance=1.0e-6):
    """ Test routine for uniform simpson integration"""
    a = 0
    b = np.pi/2
    x = np.linspace(a,b,20)
    int_estimate = simpson_integration(test_fun2,x)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_estimate-int_exact)
    working = False
    if (err<tolerance):
        print('Uniform simpson integration is OK')
        working = True
    else:
        print('Uniform simpson integration is NOT ok!!')
    return working

def test_simpson_nonuniform(tolerance=1.0e-6):
    """ Test routine for nonuniform simpson integration"""
    a = 0
    b = np.pi/2
    x = np.linspace(a,b,20)
    int_estimate = simpson_nonuniform(test_fun2,x)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_estimate-int_exact)
    working = False
    if (err<tolerance):
        print('Nonuniform simpson integration is OK')
        working = True
    else:
        print('Nonuniform simpson integration is NOT ok!!')
    return working

def test_monte_carlo_integration():
    """ 
    Test routine for monte carlo integration.
    Testing with 3*sigma error estimate, i.e., 99.7%
    similar integrations should be within this range.
    """
    a = 0
    b = np.pi/2
    blocks = 100
    iters = 1000
    int_est, err_est = monte_carlo_integration(test_fun2,a,b,blocks,iters)
    int_exact = test_fun2_int(a,b)
    err = np.abs(int_est-int_exact)
    working = False
    if (err<3.*err_est):
        print('Monte Carlo integration is OK')
        working = True
    else:
        print('Monte Carlo integration is NOT ok!!')
    return working

def test_num_grad():
    """
    Test routine for N-dimensional numerical gradient.
    """
    r = np.array([1,1,1])
    dx = np.array([0.001,0.001,0.001])
    grad_est = numerical_gradient(test_fun_numgrad,r,dx)
    grad_exact = test_fun_numgrad_grad(r)
    err = np.abs(grad_est-grad_exact)
    i = 0
    for dim in err:
        if(dim>dx[i]**2):
            print("Numerical gradient evaluation is NOT ok!")
            return False
        i+=1
    print("Numerical gradient evaluation is OK")
    return True

""" Analytical test function definitions """
def test_fun(x):
    """ This is the test function used in unit testing"""
    return np.exp(-x)

def test_fun_der(x):
    """ 
    This is the first derivative of the test 
    function used in unit testing.
    """
    return -np.exp(-x)

def test_fun_der2(x):
    """ 
    This is the second derivative of the test 
    function used in unit testing.
    """
    return np.exp(-x)

def test_fun2(x):
    """
    sin(x) in range [0,pi/2] is used for the integration tests.
    Should give 1 for the result.
    """
    return np.sin(x)

def test_fun2_int(a,b):
    """
    Integration of the test function (test_fun2).
    """
    return -np.cos(b)+np.cos(a)

def test_fun_numgrad(x):
    """
    test function given in problem 4
    """
    N = 3
    f = np.zeros(N)
    f[0] = np.sin(x[0])
    for i in range(N-2):
        f[i+1] = x[i+1]**2
    f[N-1] = np.cos(x[1])
    return f

def test_fun_numgrad_grad(r):
    grad = [np.cos(r[0]),2*r[1],-1*np.sin(r[2])]
    return grad

""" Tests performed in main """
def main():
    """ Performing all the tests related to this module """
    test_first_derivative()
    test_second_derivative()
    test_riemann_sum()
    test_trapezoid()
    test_simpson_integration()
    test_simpson_nonuniform()
    test_monte_carlo_integration()
    test_num_grad()


if __name__=="__main__":
    main()
