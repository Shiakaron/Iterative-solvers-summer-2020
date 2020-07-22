# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:18:42 2020

@author: savva
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron, lil_matrix, csc_matrix
import matplotlib.pyplot as plt

#GLOBAL variables
N_ = 10 # grid points
NN_ = N_*N_ # total number of points
p_ = 1 # set as 1 or 2
m_ = 3 # set as 3 (Van der Walls) or 4 (Casimir)
alpha_ = 0.1 
gamma_ = 0.1
epsilon_ = 0.05 # dimensionless parameter where, 0 <= epsilon << 1
lambd_ = 10 # quantifies the relative importance of electrostatic and elastic forces in the system
endl_, endr_ = -1, 1
d_ = endr_ - endl_ # domain size
dksi_ = d_/(N_-1) # deta_ = dksi_

#GLOBAL vectors, matrices
ksi = np.linspace(endl_, endr_, N_)
ksiksi, etaeta = np.meshgrid(ksi, ksi) # square grid
Ibdy = lambda:0 # information on indices (boundary, interior, corners)
M = lambda:0 # derivative matrices

def main():
    # initialise Q, Ibdy
    Q = np.reshape(0.5*ksiksi**2 + 0.5*etaeta**2, NN_) # mesh potential
    Ibdy = make_Ibdy()
    M = make_M()
    
    # ode solver
    sol = solve_ivp(ode_coupled_systems(t, y), )
    
def make_Ibdy():
    allidx = np.arange(0,NN_)
    X = np.reshape(ksiksi, NN_)
    Y = np.reshape(etaeta, NN_)
    Ibdy.All = np.nonzero((X == endr_) | (X == endl_) | (Y == endr_) | (Y == endl_))[0]
    Ibdy.Interior = np.setdiff1d(allidx, Ibdy.All)
    Ibdy.Top = np.nonzero(Y == endr_)[0]; Ibdy.Bottom = np.nonzero(Y == endl_)[0]
    Ibdy.Right = np.nonzero(X == endr_)[0]; Ibdy.Left = np.nonzero(X == endl_)[0]
    Ibdy.BottomLeft = np.intersect1d(Ibdy.Bottom, Ibdy.Left)[0]
    Ibdy.BottomRight = np.intersect1d(Ibdy.Bottom, Ibdy.Right)[0]
    Ibdy.TopLeft = np.intersect1d(Ibdy.Top, Ibdy.Left)[0]
    Ibdy.TopRight = np.intersect1d(Ibdy.Top, Ibdy.Right)[0]

def make_M():
    """
    Making derivative matrices
    For: 
        A.1 Discretising Q_ξξ, Q_ηη
        A.2 Discretising Q_ηξ, Q_ξη
    """
    # A.1 
    eye = diags([1], shape=(N_,N_))
    temp = lil_matrix(diags([-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2], shape=(N_, N_), format="csc"))
    temp[0,0]=-415/16; temp[0,1]=96; temp[0,2]=-36; temp[0,3]=32/3; temp[0,4]=-3/2
    temp[-1,-1]=-415/16; temp[-1,-2]=96; temp[-1,-3]=-36; temp[-1,-4]=32/3; temp[-1,-5]=-3/2
    temp[1,0]=10; temp[1,1]=-15; temp[1,2]=-4; temp[1,3]=14; temp[1,4]=-6; temp[1,5]=1
    temp[-2,-1]=10; temp[-2,-2]=-15; temp[-2,-3]=-4; temp[-2,-4]=14; temp[-2,-5]=-6; temp[-2,-6]=1
    temp = csc_matrix(temp/(12*dksi_*dksi_))
    M.d2Ksi = kron(eye, temp)
    M.d2eta = kron(temp, eye)
    
    # A.2
    
    
    
#     D114X = spdiags([ex -8*ex zx 8*ex -ex],-2:2,nx,nx);
# D114X(1,1) = -25; D114X(1,2) = 48; D114X(1,3) = -36; D114X(1,4)= 16; D114X(1,5)= -3;
# D114X(2,1) = -3; D114X(2,2) = -10; D114X(2,3) = 18; D114X(2,4) = -6; D114X(2,5) = 1;
# D114X(end-1,end) = 3; D114X(end-1,end-1) = 10; D114X(end-1,end-2) = -18; D114X(end-1,end-3) = 6; D114X(end-1,end-4) = -1;
# D114X(end,end) = 25; D114X(end,end-1) = -48; D114X(end,end-2) = 36; D114X(end,end-3)= -16; D114X(end,end-4)= 3;
# D114X = D114X/(12*hx);

# D114Y = spdiags([ey -8*ey zy 8*ey -ey],-2:2,ny,ny);
# D114Y(1,1) = -25; D114Y(1,2) = 48; D114Y(1,3) = -36; D114Y(1,4)= 16; D114Y(1,5)= -3;
# D114Y(2,1) = -3; D114Y(2,2) = -10; D114Y(2,3) = 18; D114Y(2,4) = -6; D114Y(2,5) = 1;
# D114Y(end-1,end) = 3; D114Y(end-1,end-1) = 10; D114Y(end-1,end-2) = -18; D114Y(end-1,end-3) = 6; D114Y(end-1,end-4) = -1;
# D114Y(end,end) = 25; D114Y(end,end-1) = -48; D114Y(end,end-2) = 36; D114Y(end,end-3)= -16; D114Y(end,end-4)= 3;
# D114Y = D114Y/(12*hy);
# M.D2XY4 = kron(D114Y,D114X);M.D1XC = kron(Iy,D114X);M.D1YC = kron(D114Y,Ix);
    
    
    
    

def ode_coupled_systems(t, y):
    """
    y[0] = Q
    y[1] = u
    """

  
def compute_g(u):
    if epsilon == 0:
        return min((1+u)**3)
    else:
        return 1
    
        
        
if (__name__ == "__main__"):
    main()
