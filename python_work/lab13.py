# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:30:15 2020

@author: savva
"""


import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    """
    n = 50
    k = 1e-2
    nsteps = 1000
    x = np.linspace(0,1,n+1)
    u0 = np.zeros(n+1)
    # f = lambda t: np.sin(10*t)/5
    f = lambda t: 2*np.exp(-t)*np.sin(10*t)/5
    
    # initialise
    U = np.zeros([nsteps+1,n+1])
    U[0] = u0
    r = np.arange(1,n,1)
    a = 1 + k/2
    b = 1 - k/2
    c = k**2*n**2
    
    # for plotting
    plt.close("all")
    fig, ax = plt.subplots()
    ln = ax.plot(x,U[0],"o-")
    plt.xlabel('x')
    plt.ylabel('u')
    plt.ylim(-0.4,0.4)
    
    # s = 0
    U[1,0] = f(k)
    # s = 1:nsteps
    for s in range(nsteps):
        U[s+1,r] = (2*U[s,r] - b*U[s-1,r] + c*(U[s,r+1] - 2*U[s,r] + U[s,r-1]))/a
        U[s+1,0] = f(s*k)
        # draw
        ln[0].set_ydata(U[s+1])
        plt.title('t = {:1.3f}'.format((s+1)*k))
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    
if (__name__ == "__main__"):
    main()