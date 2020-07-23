# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:18:42 2020

@author: savva
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron, csc_matrix
from scipy.fft import dct, idct
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
Q = lambda:0 # mesh potential and all its derivatives
u = lambda:0 # solution and its derivatives
monitor = None # monitor function
J = None # Hessian of Q


def main():
    global Q, Ibdy, M
    # initialise Q, Ibdy, M, u
    Q.val = np.reshape(0.5*ksiksi**2 + 0.5*etaeta**2, NN_) # mesh potential
    make_Ibdy()
    make_M()
    u.val = np.zeros(NN_) # initialise u
    
    # ode solver
    # sol = solve_ivp(ode_coupled_systems(t, y), )
    
def make_Ibdy():
    """
    Making arrays containg indices information
    """
    allidx = np.arange(0,NN_)
    X = np.reshape(ksiksi, NN_)
    Y = np.reshape(etaeta, NN_)
    Ibdy.Boundary = np.nonzero((X == endr_) | (X == endl_) | (Y == endr_) | (Y == endl_))[0]
    Ibdy.Interior = np.setdiff1d(allidx, Ibdy.Boundary)
    Ibdy.Top = np.nonzero(Y == endr_)[0]; Ibdy.Bottom = np.nonzero(Y == endl_)[0]
    Ibdy.Right = np.nonzero(X == endr_)[0]; Ibdy.Left = np.nonzero(X == endl_)[0]
    Ibdy.BottomLeft = np.intersect1d(Ibdy.Bottom, Ibdy.Left)[0]
    Ibdy.BottomRight = np.intersect1d(Ibdy.Bottom, Ibdy.Right)[0]
    Ibdy.TopLeft = np.intersect1d(Ibdy.Top, Ibdy.Left)[0]
    Ibdy.TopRight = np.intersect1d(Ibdy.Top, Ibdy.Right)[0]

def make_M():
    """
    Making derivative matrices                
    """
    eye = diags([1], shape=(N_,N_))
    # A.1 
    temp = diags([-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2], shape=(N_, N_), format="lil")
    temp[0,0]=-415/6; temp[0,1]=96; temp[0,2]=-36; temp[0,3]=32/3; temp[0,4]=-3/2
    temp[-1,-1]=-415/6; temp[-1,-2]=96; temp[-1,-3]=-36; temp[-1,-4]=32/3; temp[-1,-5]=-3/2
    temp[1,0]=10; temp[1,1]=-15; temp[1,2]=-4; temp[1,3]=14; temp[1,4]=-6; temp[1,5]=1
    temp[-2,-1]=10; temp[-2,-2]=-15; temp[-2,-3]=-4; temp[-2,-4]=14; temp[-2,-5]=-6; temp[-2,-6]=1
    temp = csc_matrix(temp/(12*dksi_*dksi_))
    M.d2ksi = kron(eye, temp) # d^2f/dksi^2
    M.d2eta = kron(temp, eye) # d^2f/deta^2
    
    # A.2
    temp = diags([1, -8, 8, -1], [-2, -1, 1, 2], shape=(N_, N_), format="lil")
    temp[0,0]=-25; temp[0,1]=48; temp[0,2]=-36; temp[0,3]=16; temp[0,4]=-3
    temp[-1,-1]=25; temp[-1,-2]=-48; temp[-1,-3]=36; temp[-1,-4]=-16; temp[-1,-5]=3
    temp[1,0]=-3; temp[1,1]=-10; temp[1,2]=18; temp[1,3]=-6; temp[1,4]=1
    temp[-2,-1]=3; temp[-2,-2]=10; temp[-2,-3]=-18; temp[-2,-4]=6; temp[-2,-5]=-1
    temp = csc_matrix(temp/(12*dksi_))
    M.dksiCentre = kron(eye,temp) # df/dksi centre difference
    M.detaCentre = kron(temp,eye) # df/deta centre difference
    M.dksideta = kron(temp,temp) #d^2f/dksideta
    
    # C - upwinding scheme
    #forward
    temp = diags([-3,4,-1], [0,1,2], shape=(N_,N_), format="lil")
    temp[-1,-1]=3; temp[-1,-2]=-4; temp[-1,-3]=1
    temp[-2,-1]=2; temp[-2,-2]=-2
    temp - csc_matrix(temp/(2*dksi_))
    M.dksiForw = kron(eye, temp) # df/dksi forward difference
    M.detaForw = kron(temp, eye) # df/deta forward difference
    
    #backward
    temp = diags([1,-4,3], [-2,1,0], shape=(N_,N_), format="lil")
    temp[0,0]=-3; temp[0,1]=4; temp[0,2]=-1
    temp[1,0]=-2; temp[1,1]=2
    temp - csc_matrix(temp/(2*dksi_))
    M.dksiBack = kron(eye, temp) # df/dksi backward difference
    M.detaBack = kron(temp, eye) # df/deta backward difference
    
    # for discreet cosine transform
    temp = (2*np.cos(np.pi*np.arange(0,N_)/N_)-2).reshape((N_,1))*np.ones(N_) + \
    np.ones((N_,1))*(2*np.cos(np.pi*np.arange(0,N_)/N_)-2)
    M.Leig = temp/(dksi_*dksi_)
    
    # Smoothing Matrix
    off1 = np.ones(NN_-1)
    off1[(N_-1)::N_] = 0 
    off2 = np.ones(NN_-3)
    off2[0::N_] = 0 
    M.Sm = diags([off1[:-N_],2,off2,2*off1,4,2*off1,off2,2,off1[:-N_]], [-N_-1,-N_,-N_+1,-1,0,1,N_-1,N_,N_+1], shape=(NN_, NN_))/16

def compute_Q_spatial_ders():
    """
    compute spacial (ksi, eta) derivatives of the mesh potetial
    """
    # 1st derivatives
    Q.dksi = M.dksiCentre*Q.val
    Q.deta = M.detaCentre*Q.val
    # 2nd derivatives
    extra = 25/(6*dksi_)
    temp = np.zeros(NN_); temp[Ibdy.Left] = extra; temp[Ibdy.Right] = extra
    Q.d2ksi = M.d2ksi*Q.val + temp
    temp = np.zeros(NN_); temp[Ibdy.Top] = extra; temp[Ibdy.Bottom] = extra
    Q.d2eta = M.d2eta*Q.val + temp
    Q.dksideta = M.dksideta*Q.val; Q.dksideta[Ibdy.Boundary] = 0

def compute_u_spatial_ders():
    """
    compute spatial (x, y) derivatives of the solution
    """
    
    
    
def compute_monitor():
    """
    computes the monitor functino Mon
    for epsilon == 0
        Mon = 1/(1+u)^6
    for epsilon > 0
        for p = 1
            Mon = 1 + (u_x)^2 + (u_y)^2
        for p = 2
            Mon = |u_xx + u_yy|^2
    """
    

def solve_PMA():
    """
    solve for dQdt
    """
    global J
    compute_u_spatial_ders()
    compute_monitor()
    J = np.multiply(Q.d2ksi, Q.d2eta) - np.multiply(Q.dksideta, Q.dksideta)
    q_rhs = np.sqrt(np.multiply(monitor, np.abs(J)))/epsilon_
    temp = dct(q_rhs.reshape(N_,N_))
    dQdt = idct(np.divide(temp,(1-gamma_*M.Leig)))
    return dQdt.reshape(NN_)


def ode_coupled_systems(t, y):
    """
    y[0] = Q
    y[1] = u
    exluded: y[2] = t'
    """
    
    


    #return [dQdt, dudt] 

  
def compute_g(u):
    if epsilon_ == 0:
        return min((1+u)**3)
    else:
        return 1
    
        
        
if (__name__ == "__main__"):
    main()
