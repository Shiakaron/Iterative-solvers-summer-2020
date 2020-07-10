# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:46:30 2020

@author: savva
"""

import math
import numpy as np
from scipy.sparse import diags, block_diag, linalg
from scipy.optimize import newton_krylov
import matplotlib.pyplot as plt
import matplotlib


d = 40 # domain size
N = 64  # points in each direction
h = d/N # mesh spacing
k = 0.2 # time step
Tf = 500 # final time

# random initial condition
U = np.random.randn(N**2) # this corresponds to U[0]
Uo = None # this is U[0-1] which is set equal to U[0] for simplicity
UoUo = None
UoUoUo = None

# pde parameters
r = 0.2
g = 0

# FORM THE DISCREET LAPLACIAN
e = 1/h**2
nn = N**2
A = np.tile(diags([e, e, -4*e, e, e], [1-N,-1,0,1,N-1], shape=(N, N), format="csc"),N)
Lap = block_diag(A[:]) + diags([e,e,e,e], [N-nn,-N,N,nn-N], shape=(nn, nn), format="csc")
    
# FORM THE DISCRETE LINEAR LAPLACIAN
I = diags([1],[0],shape=(nn,nn),format="csc")
L = -Lap*Lap - 2*Lap + (r-1)*I

# for plotting
fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(U.reshape(N,N), extent=[0,d,0,d], interpolation="sinc")
plt.title("0")
plt.tight_layout()

def residual(u):
    uu = np.multiply(u,u)
    return (u-Uo)/k - (L @ u + g*uu - np.multiply(u,uu) + L @ Uo + g*UoUo - UoUoUo)/2
 
Nsteps = math.ceil(Tf/k);    # ESTIMATE NUMBER OF TIME STEPS REQUIRED
plotSteps = round(1/k); # PLOT ROUGHLY EVERY ONE TIME UNIT
for s in range(Nsteps):
    # HERE U is U[s] and Uo is U[s-1]
    # BEFORE OVERWRITING U, STORE U[s] to Uo for the next iteration
    Uo = U.copy(); 
    UoUo = np.multiply(Uo,Uo)
    UoUoUo = np.multiply(Uo,UoUo)
    
    # NEWTON KRYLOV
    U = newton_krylov(residual, Uo, verbose=1)

    # plot
    if s%plotSteps==0:
        plt.title('t = {:1.3f}'.format((s+1)*k)) 
        im.set_data(U.reshape(N,N))
        im.set_clim(vmax=max(U),vmin=min(U))
        fig.canvas.draw()
        fig.canvas.flush_events()


