# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:08:20 2020

@author: savva
"""
# SDS401/COS502
# LAB 12 - Heat equation with homogeneous Dirichlet BCs

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib qt    # For people running on Jupyter notebooks

# Solver options (both solver and solver_plot)
#   n: number of intervals (n+1 nodes)
#   a: diffusion coefficient
#   k: time step
#   nsteps: number of steps to take
#   u0: initial condition
#
#   For solver_plot only
#       show : False or True to not show or show results
#       exact: provide name of lambda function for exact solution
# 
# Running "solver" and "solver_plot" with show=False is actually identiical;
#   solver is included to demonstrate the simplicity of the code. 
def solver(n,k,nsteps,u0,a=1):
    p = a*k*n**2                 # scheme parameter
    U = np.zeros([nsteps+1,n+1]) # solution array
    U[0] = u0	                 # initial condition

    for s in range(nsteps):
        U[s+1,1:-1] = p*U[s,2:] + (1-2*p)*U[s,1:-1] + p*U[s,:-2]
    
    return U
    
def solver_plot(n,k,nsteps,u0,a=1,show=False,exact=None):    
    p = a*k*n**2                 # scheme parameter, n=1/h
    U = np.zeros([nsteps+1,n+1]) # solution array
    U[0] = u0	                 # initial condition
    print("p = ", p)
    if p>=0.5:
        print("p greater than 1/2. This can be unstable")
    if show:  # Prepare plotting window and plot the IC 
        plt.close('all')
        fig, ax = plt.subplots()
        
        if exact is not None:
            ln = ax.plot(x, U[0], 'o-',x,exact(x,0))
        else:
            ln = ax.plot(x,U[0],'o-')
            
        plt.xlabel('x')
        plt.ylabel('u')
            
    for s in range(nsteps):
        # all except first and last node
        U[s+1,1:-1] = p*U[s,2:] + (1-2*p)*U[s,1:-1] + p*U[s,:-2]
        # first node. insulation implies the first x derivative is zero(?)
        U[s+1,0] = 2*p*U[s,1] + (1-2*p)*U[s,0] 
        
        if show:    # Update the y data of the curves plotted
            ln[0].set_ydata(U[s+1])
            
            if exact is not None:
                ln[1].set_ydata(exact(x,(s+1)*k))                
            
            # Force drawing
            plt.title('t = {:1.3f}'.format((s+1)*k))
            fig.canvas.draw()
            fig.canvas.flush_events()
    return x,U

##################################################################
n = 23
k = 1e-3
a = 0.5
nsteps = 1000	
show = True
x = np.linspace(0,1,n+1)     # nodes of the domain
u0 = np.cos(np.pi*x/2)         # initial condition

# Plot the solution when t=nsteps*k and compare with exact
u = lambda x,t: np.exp(-a*np.pi**2*t/4)*np.cos(np.pi*x/2)

if show:
    x,U = solver_plot(n,k,nsteps,u0,a,show=True,exact=u)
else:
    U = solver(n,k,nsteps,u0,a)
    
    plt.clf()
    plt.plot(x,u(x,nsteps*k),label='exact')
    plt.plot(x,U[-1],'o')
    plt.legend()



