"""
This module contains functions for numerical calculus:
- first and second derivatives
- 1D integrals: Riemann, trapezoid, Simpson, 
  and Monte Carlo with uniform random numbers
"""

import numpy as np

def eval_derivative(function, x, dx ):
    """ 
    This calculates the first derivative with
    symmetric two point formula, which has O(h^2)
    accuracy. See, e.g., FYS-4096 lecture notes.
    """
    return (function(x+dx)-function(x-dx))/2/dx

def eval_2nd_derivative(function, x, dx):
    """ 
    This calculates the second derivative with
    O(h^2) accuracy. See, e.g., FYS-4096 lecture 
    notes.
    """
    return (function(x+dx)+function(x-dx)-2.*function(x))/dx**2

def eval_partial_derivative(func,x,dx,dim):
    """ 
    This calculates the first partial derivative 
    of f(x) along dimension dim, where x is a vector. 
    The derivative uses symmetric two point formula, 
    which has O(h^2) accuracy. See, e.g., FYS-4096 
    lecture notes.
    """
    h=dx[dim]
    dx=0.0*dx
    dx[dim]=h
    return (func(x+dx)-func(x-dx))/2/h

def gradient(func,x,h):
    """ 
    This calculates the gradient of f(x) along dimension 
    dim, where x is a vector. The derivatives are calculated
    with functions eval_derivative (for 1D) and 
    eval_partial_derivative (for dims>1). Currently both
    routines use symmetric two point formula, 
    which has O(h^2) accuracy. See, e.g., FYS-4096 
    lecture notes.
    """
    if np.isscalar(x):
        grad=eval_derivative(func,x,h)
    else:
        grad=np.zeros((len(x),))
        for i in range(len(x)):
            grad[i]=eval_partial_derivative(func,x,h,i)
    return grad

def riemann_sum(x,function):
    """ 
    Left Rieman sum for uniform grid. 
    See, e.g., FYS-4096 lecture notes.
    """
    dx=x[1]-x[0]
    f=function(x)
    return np.sum(f[0:-1])*dx

def trapezoid(x,function):
    """ 
    Trapezoid for uniform grid. 
    See, e.g., FYS-4096 lecture notes.
    """
    dx=x[1]-x[0]
    f=function(x)
    return (f[0]/2+np.sum(f[1:-1])+f[-1]/2)*dx

def simpson_integration(x,function):
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

def simpson_nonuniform(x,function):
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



""" Test routines for unit testing """
def test_first_derivative(tolerance=1.0e-3):
    """ Test routine for first derivative of f"""
    x = 0.8
    dx = 0.01
    df_estimate = eval_derivative(test_fun,x,dx)
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
    df_estimate = eval_2nd_derivative(test_fun,x,dx)
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
    int_estimate = riemann_sum(x,test_fun2)
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
    int_estimate = trapezoid(x,test_fun2)
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
    x = np.linspace(a,b,100)
    int_estimate = simpson_integration(x,test_fun2)
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
    x = np.linspace(a,b,100)
    int_estimate = simpson_nonuniform(x,test_fun2)
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

if __name__=="__main__":
    main()
