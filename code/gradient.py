"""
gradient.py -- gradient descent in Python
"""

import numpy as np
import numpy.linalg as la

def graddesc(A, b, c, x, tol):
    """
    Compute gradient descent for function x'*A*x + b'*x + c with A psd
    """
    # Compute the gradient
    grad = np.dot(A,x)+b
    # Start with the initial point
    xout = [x]
    xold = x+2*tol*np.ones(2)
    while la.norm(x-xold,2) > tol:
        # If the gradient is bigger than the tolerance
        xold = x
        Agrad = np.dot(A,grad)
        alpha = np.dot(grad,grad)/np.dot(grad,Agrad)
        x = x - alpha*grad
        xout.append(x)
        grad = grad-alpha*Agrad
    return np.array(xout).transpose()