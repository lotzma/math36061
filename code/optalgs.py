"""
optalgs.py - Hand-made optimization algorithms
"""

import numpy as np
import numpy.linalg as la

def graddesc_bt(f, df, x0, tol, maxiter=100, rho=0.5, c=0.1):
    """
    Gradient descent with backtracking
    """
    x, xold = x0, x0+2*tol*np.ones(x0.shape)
    iter = 0
    while ( la.norm(x-xold) > tol ) and ( iter < maxiter ):
        xold = x
        p = -df(x)
        # Do backtracking
        while f(x+c*alpha*p) <= f(x) + c*alpha*np.dot(p, grad):
            alpha = alpha*rho
        xold = x
        x += c*alpha*p
    return x, iter

def graddesc_fx(f, df, x0, tol, maxiter=100, alpha=0.1):
    """
    Gradient descent with fixed step length
    """
    x, xold = x0, x0+2*tol*np.ones(x0.shape)
    iter = 0
    while ( la.norm(x-xold) > tol ) and ( iter < maxiter ):
        # Find step length
        p = -df(x)
        xold = x
        x += alpha*p
    return x, iter
            
def newton(f, df, ddf, x0, tol, maxiter=100):
    """
    Newton's method
    """
    x, xold = x0, x0+2*tol*np.ones(x0.shape)
    iter = 0
    while ( la.norm(x-xold) > tol ) and ( iter < maxiter ):
        grad = df(x)
        hess = ddf(x)
        z = la.solve(hess,grad)[0]
        xold = x
        x = x-z
        iter += 1
    return x, iter

def bfgs(f, df, B, x0, tol, maxiter=100):
    pass