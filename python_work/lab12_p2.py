# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:36:59 2020

@author: savva
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("start")
    a = 1
    n = 10
    k = 0.025
    nsteps = 15
    x = np.linspace(0,1,n+1) # spatial nodes
    u0 = np.sin(np.pi*x) # initial condition
    
    # initialise 
    p = a*k*n**2 
    U = np.zeros([nsteps+1,n+1])
    U[0] = u0 
    r = np.arange(1,n,1) # indices of interior nodes   
    
    # for plotting
    plt.close("all")
    fig, ax = plt.subplots()
    ln = ax.plot(x,U[0],"o-")
    plt.xlabel('x')
    plt.ylabel('u')
    
    # LU factorisation
    alph = -p*np.ones(n-1)
    bet = 2*(p+1)*np.ones(n)
    l, v = trilu(alph, bet, alph) 
    
    # time steps
    for s in range(nsteps):
        print(s, U[s])
        b = p*U[s,r+1] + (1-2*p)*U[s,r] + p*U[s,r-1]
        # Dirichlet boundary conditions: u(0,t) = u(1,t) = 0
        U[s+1] = np.append(0, np.append(trisolve(l, v, alph, b), 0))
        
        # draw
        ln[0].set_ydata(U[s+1])
        plt.title('t = {:1.3f}'.format((s+1)*k))
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    print("finished")

def trilu(alpha,beta,gamma):
    """
    Parameters
    ----------
    a : lower diagonal
    b : diagonal
    c : upper diagonal

    Returns
    -------
    l : lower diagonal of matrix L
    v : upper diagonal of matrix U
    
    such that LU = A

    """
    n = len(beta)
    l = np.zeros(n-1)
    v = np.zeros(n)
    
    v[0] = beta[0]
    for i in range(1,n):
        l[i-1] = alpha[i-1]/v[i-1]
        v[i] = beta[i]-l[i-1]*gamma[i-1]

    return l, v

def trisolve(l, v, gamma, b):
    n = len(b)
    y = np.zeros(n)
    u = np.zeros(n)
    
    # forward solve for y
    y[0]=b[0]
    for i in range(1,n):
        y[i] = b[i] - l[i-1]*y[i-1]
    
    # backward solve for u
    u[-1]=y[-1]/v[-1]
    for i in reversed(range(n-1)):
        u[i] = (y[i]-gamma[i]*u[i+1])/v[i]
    
    return u

if (__name__ == "__main__"):
    main()