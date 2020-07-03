# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:19:25 2020

@author: savva

SDS401/502 LAB14: Wave equation in a 2D tank
Problem 1
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib qt    # For people running on Jupyter notebooks

f = lambda x,y: -.4*np.exp(-100*(x-0.3)**2-100*(y-0.4)**2)

Tf = 10                   # Final time
c = 1                     # wave speed
k = 1e-2                  # time step
p = 0.4                   # p parameter
n = int(np.sqrt(p)/c/k)   # number of mesh points
nsteps=int(round(Tf/k))	  # number of time steps to take
xx = np.arange(n+1)/n     
x,y=np.meshgrid(xx,xx)    # Mesh for x and y variables

L = np.zeros([n+1,n+1])   # Laplacian; not needed if storage is crucial
i = np.arange(1,n)        # Indices for interior nodes in x direction
j = np.arange(1,n)        # Indices for interior nodes in y direction
u0= f(x,y)                # Initial condition - solution at t=s*k
up= np.zeros([n+1,n+1])   # Solution at t=(s-1)*k
u = u0.copy()             # Solution at t=(s+1)*k
gamma = 4

# Settings for animation
fig = plt.figure()
ax = fig.gca(projection='3d')
surf =ax.plot_surface(x, y, u0, cmap='viridis')
ax.view_init(elev=30, azim=-120)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
ax.set_zlim3d(-.2,.2)  
ls = LightSource(azdeg=0, altdeg=65)

# Loop for time stepping
for s in range(nsteps):
    # Interior points	
    L[1:-1,1:-1]= p*(u[2:,1:-1]+u[1:-1,2:] -4*u[1:-1,1:-1] 
                    + u[0:-2,1:-1]+u[1:-1,0:-2])

    # Boundary conditions
    L[0,j] = p*(2*u[1,j]+u[0,j+1]-4*u[0,j]+u[0,j-1])   # i=0, left
    L[n,j] = p*(u[n,j+1]-4*u[n,j]+2*u[n-1,j]+u[n,j-1]) # i=n, right
    L[i,0] = p*(u[i+1,0]+2*u[i,1]-4*u[i,0]+u[i-1,0])   # j=0, bottom
    L[i,n] = p*(u[i+1,n]-4*u[i,n]+u[i-1,n]+2*u[i,n-1]) # j=n, top

    if s==0: # Apply zero velocity condition
        u = u0 + 0.5*L
    else:    # Apply PDE
        u = (2*u0 - (1-0.5*gamma*k)*up  +  L)/(1+0.5*gamma*k) 
 
    # Corners - not really captured by the method
    # Alternatively, we could have set u[0,0] = np.nan etc.
    u[0,0] = 0.5*(u[0,1]+u[1,0])
    u[0,n] = 0.5*(u[0,n-1]+u[1,n])
    u[n,0] = 0.5*(u[n-1,0]+u[n,1])
    u[n,n] = 0.5*(u[n-1,n]+u[n,n-1])    
    
    # Prepare arrays for next iteration    
    up = np.array(u0)
    u0 = np.array(u)    

    # Update plots
    surf.remove() 
    surf = ax.plot_surface(x, y, u,color=(0,0.8,1),rstride=1, cstride=1, linewidth=0,
                       antialiased=False,alpha=0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()      