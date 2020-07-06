# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:42:36 2020

@author: savva
"""

"""
SDS401/502 LAB14: 2D POISSON EQUATION
Problem 2
"""
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Number of intervals per direction
n = 1000

# Mesh spacing and number of unknown nodes
h = 1/n
N = n*(n-1)

# Diagonals of Laplacian operator
main  = -4*np.ones(N)
off = np.ones(N-1)
off[n-2::n-1] = 0

upper = np.ones((n-1)**2)
lower = np.ones((n-1)**2)
lower [1-n:] = 2
 
# Form the matrix
A = scipy.sparse.diags(diagonals=[main,off,off,upper,lower],\
                       offsets=[0,1,-1,n-1,1-n],format='csc')

# Right-hand-side vector
b = -h**2*np.ones(N)

# Solve the system
uu = scipy.sparse.linalg.spsolve(A,b)

# 2D arrays for plotting
u = np.zeros([n+1,n+1])
u[1:,1:-1] = np.reshape(uu,(n,n-1))

# Flow Rate (computed from 2D Trapezium Rule)
Q = 0.25*h**2*(4*np.sum(u)-2*np.sum(u[0,:]+u[-1,:]+u[:,0]+u[:,-1]) \
     +u[0,0]+u[-1,0]+u[-1,-1]+u[0,-1])

# Exact Solution (from method of separation of variables)
M = 1600 # Number of terms in the series
m = np.arange(1,2*M,2)
Q_exact = sum(8*(m*np.pi-np.tanh(m*np.pi))/(np.pi*m)**5) # The analytical sum

# Print some stats
print('Numerical :'+'{:1.11f}'.format(Q))
print('Exact     :'+'{:1.11f}'.format(Q_exact))
print('Error (\%):'+'{:1.2e}'.format(100-100*Q/Q_exact))

# Plot solution
xx = np.linspace(0,1,num=n+1)
x,y = np.meshgrid(xx,xx)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, u, cmap='viridis')
ax.view_init(elev=30, azim=-120)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
ax.set_title('n = ' + str(n))
plt.tight_layout()
