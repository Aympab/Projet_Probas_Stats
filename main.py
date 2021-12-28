from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from math import exp

def cov_k(x0, x1, lambd = 1) :
    return (1+ ((x0 - x1)/lambd) + (((x0-x1)**2)/(3*(lambd**2)))) * exp(-((x0-x1)/lambd))

def compute_cov_sigma(x, lambd = 1) :
    return np.array([[cov_k(xi, xj, lambd) for xj in x] for xi in x])


def compute_sigma(x, lambd = 1) :
    sigma = np.zeros((x.shape[0], x.shape[0]))
    for i, x0 in enumerate(x) :
        for j, x1 in enumerate(x) :
            sigma[i, j] = abs(cov_k(x0, x1, lambd=lambd))
    
    return np.tril(sigma)



if __name__ == '__main__':
    N = 1000

    _out = True
    if N > 50 :
        _out = False

    lambd = 1

    #To get N numbers evenly spread between 0 and 1, linspace() from numpy
    xi = np.linspace(0, 1, N, dtype=np.double)
    sigma = compute_cov_sigma(xi)

    if(_out) :
        print("xi : ", xi.round(2))
        print("Sigma :")
        print(sigma.round(3))
    else:
        print("xi and Sigma computed !")
        
        
    L = la.cholesky(sigma)

    G = np.random.normal(0, 1, N)

    m = xi

    if _out :
        print("L :\n", L.round(2))
        print("G :\n", G.round(2))
        print("m :\n", m.round(2))
    else :
        print("L, G and m computed !")
        
        
        
        
    sig2 = compute_sigma(xi)

    i_upper = np.triu_indices(N, 0)
    sig2[i_upper] = sig2.T[i_upper] 

    M = sig2 - sigma
    print(la.norm(M))
    print(sig2.round(2))