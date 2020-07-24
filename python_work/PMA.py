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
dksi2_ = dksi_*dksi_

#GLOBAL vectors, matrices
ksi = np.linspace(endl_, endr_, N_)
ksiksi, etaeta = np.meshgrid(ksi, ksi) # square grid
Ibdy = lambda:0 # information on indices (boundary, interior, corners)
M = lambda:0 # derivative matrices
Q = lambda:0 # mesh potential and all its derivatives
u = lambda:0 # solution and its derivatives
monitor = None # monitor function
J = None # Hessian (Jacobian) of Q


def main():
    global Q, Ibdy, M
    # initialise Q, Ibdy, M, u
    Q.val = np.reshape(0.5*ksiksi**2 + 0.5*etaeta**2, NN_) # mesh potential
    make_Ibdy()
    make_M()
    u.val = np.zeros(NN_) # initialise u
    
    # each time step:
    solve_PMA()
    
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
    temp[0,:5] = [-415/6, 96, -36, 32/3, -1.5]
    temp[1,:6] = [10, -15, -4, 14, -6, 1]
    temp[-1,-5:] = [-1.5, 32/3, -36, 96, -415/6]
    temp[-2,-6:] = [1, -6, 14, -4, -15, 10]
    temp = csc_matrix(temp/(12*dksi2_))
    M.d2ksi = kron(eye, temp) # d^2f/dksi^2
    M.d2eta = kron(temp, eye) # d^2f/deta^2
    
    # A.2
    temp = diags([1, -8, 8, -1], [-2, -1, 1, 2], shape=(N_, N_), format="lil")
    temp[:2,:5] = [[-25, 48, -36, 16, -3], [-3, -10, 18, -6, 1]]
    temp[-2:,-5:] = [[-1, 6, -18, 10, 3], [3, -16, 36, -48, 25]]
    temp = csc_matrix(temp/(12*dksi_))
    M.dksiCentre = kron(eye,temp) # df/dksi centre difference
    M.detaCentre = kron(temp,eye) # df/deta centre difference
    M.dksideta = kron(temp,temp) #d^2f/dksideta
    
    # C - upwinding scheme
    #forward
    temp = diags([-3,4,-1], [0,1,2], shape=(N_,N_), format="lil")
    temp[-1,-3:] = [1, -4, 3]
    temp[-2,-2:] = [-2, 2]
    temp - csc_matrix(temp/(2*dksi_))
    M.dksiForw = kron(eye, temp) # df/dksi forward difference
    M.detaForw = kron(temp, eye) # df/deta forward difference
    
    #backward
    temp = diags([1,-4,3], [-2,1,0], shape=(N_,N_), format="lil")
    temp[0,:3] = [-3, 4, -1]
    temp[1,:2] = [-2, 2]
    temp - csc_matrix(temp/(2*dksi_))
    M.dksiBack = kron(eye, temp) # df/dksi backward difference
    M.detaBack = kron(temp, eye) # df/deta backward difference
    
    # for discreet cosine transform
    temp = (2*np.cos(np.pi*np.arange(0,N_)/N_)-2).reshape((N_,1))*np.ones(N_) + \
    np.ones((N_,1))*(2*np.cos(np.pi*np.arange(0,N_)/N_)-2)
    M.Leig = temp/dksi2_
    
    # Smoothing Matrix
    off1 = np.ones(NN_-1)
    off1[(N_-1)::N_] = 0 
    off2 = np.ones(NN_-3)
    off2[0::N_] = 0 
    M.Sm = diags([off1[:-N_],2,off2,2*off1,4,2*off1,off2,2,off1[:-N_]], \
                 [-N_-1,-N_,-N_+1,-1,0,1,N_-1,N_,N_+1], shape=(NN_, NN_))/16

def compute_Q_spatial_ders():
    """
    compute spatial (ksi, eta) derivatives of the mesh potetial
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
    # 1st derivatives (ksi, eta) - not saved in u struct
    u_dksi = M.dksiCentre*u.val
    u_deta = M.detaCentre*u.val
    # 1st derivatives (x, y)
    u.dx = np.divide(np.multiply(Q.d2eta,u_dksi) - np.multiply(Q.dksideta,u_deta), J)
    u.dy = np.divide(- np.multiply(Q.dksideta,u_dksi) + np.multiply(Q.d2ksi,u_deta), J)
    # 2nd derivatives (xx, yy) - laplacian operator
    u.xx, u.yy = Laplace_operator(np.reshape(u.val,(N_,N_)), u_dksi, u_deta)
    
def Laplace_operator(v, v_dksi, v_deta):
    """    
    L(v) = v_xx + v_yy = J^-1 * div_ksi { J^-1 * A * grad_ksi (v) }
    more consicely:
    v_xx = J^-1 * [ ( A11 * v_dksi )_ksi + ( A12 * v_deta)_ksi ]
    v_yy = J^-1 * [ ( A12 * v_dksi )_eta + ( A22 * v_deta)_eta ]
    where appendix B shows the discretisation of the bracketed terms.
       
    The argument v needs to be an NxN array
    v_dksi and v_deta are the ksi and eta derivatives of v and need to be 1D arrays of size NN_
    """    
    # initialise 
    A11 = np.reshape(np.divide(Q.dksideta**2 + Q.d2eta**2, J), (N_, N_))
    A22 = np.reshape(np.divide(Q.dksideta**2 + Q.d2ksi**2, J), (N_, N_))
    A12 = -np.divide(np.multiply(Q.dksideta, Q.d2ksi + Q.d2eta), J)
    v_xx = np.zeros((N_, N_)); v_yy = np.zeros((N_, N_))
    
    # B.1 : (A11*u_dksi)_ksi, (A22*u_deta)_eta
    # interior points 
    r = np.arange(3,N_-3) 
    v_xx[:,r] = (4*np.multiply(A11[:,r-1], (v[:,r-3] - 8*v[:,r-2] + 8*v[:,r] - v[:,r+1])) 
                  -np.multiply((-A11[:,r-2] + 9*A11[:,r-1] + 9*A11[:,r] - A11[:,r+1]), 
                               (v[:,r-2] - 27*v[:,r-1] + 27*v[:,r] - v[:,r+1])) 
                  +np.multiply((-A11[:,r-1] + 9*A11[:,r] + 9*A11[:,r+1] - A11[:,r+2]), 
                               (v[:,r-1] - 27*v[:,r] + 27*v[:,r+1] - v[:,r+2]))
                  -4*np.multiply(A11[:,r+1], (v[:,r-1] - 8*v[:,r] + 8*v[:,r+2] - v[:,r+3])))/(288*dksi2_)
    
    v_yy[r,:] = (4*np.multiply(A22[r-1,:], (v[r-3,:] - 8*v[r-2,:] + 8*v[r,:] - v[r+1,:]))
                  -np.multiply((-A22[r-2,:] + 9*A22[r-1,:] + 9*A22[r,:] - A22[r+1,:]), 
                               (v[r-2,:] - 27*v[r-1,:] + 27*v[r,:] - v[r+1,:]))
                  +np.multiply((-A22[r-1,:] + 9*A22[r,:] + 9*A22[r+1,:] - A22[r+2,:]), 
                               (v[r-1,:] - 27*v[r,:] + 27*v[r+1,:] - v[r+2,:]))
                  -4*np.multiply(A22[r+1,:], (v[r-1,:] - 8*v[r,:] + 8*v[r+2,:] - v[r+3,:])))/(288*dksi2_)
    
    # next-to boundary points
    v_xx[:,1] = np.multiply(A11[:,1], (10*v[:,0] - 15*v[:,1] - 4*v[:,2] + 14*v[:,3] - 6*v[:,4] + v[:,5]))/(12*dksi2_) \
        + np.multiply((-3*v[:,0] - 10*v[:,1] + 18*v[:,2] - 6*v[:,3] + v[:,4]), 
                      (-3*A11[:,0] - 10*A11[:,1] + 18*A11[:,2] - 6*A11[:,3] + A11[:,4]))/(144*dksi2_)
        
    v_yy[1,:] = np.multiply(A22[1,:], (10*v[0,:] - 15*v[1,:] - 4*v[2,:] + 14*v[3,:] - 6*v[4,:] + v[5,:]))/(12*dksi2_) \
        + np.multiply((-3*v[0,:] - 10*v[1,:] + 18*v[2,:] - 6*v[3,:] + v[4,:]), 
                      (-3*A22[0,:] - 10*A22[1,:] + 18*A22[2,:] - 6*A22[3,:] + A22[4,:]))/(144*dksi2_)
    
    v_xx[:,-2] = np.multiply(A11[:,-2], (10*v[:,-1] - 15*v[:,-2] - 4*v[:,-3] + 14*v[:,-4] - 6*v[:,-5] + v[:,-6]))/(12*dksi2_) \
        + np.multiply((-3*v[:,-1] - 10*v[:,-2] + 18*v[:,-3] - 6*v[:,-4] + v[:,-5]), 
                      (-3*A11[:,-1] - 10*A11[:,-2] + 18*A11[:,-3] - 6*A11[:,-4] + A11[:,-5]))/(144*dksi2_)
        
    v_yy[-2,:] = np.multiply(A22[-2,:], (10*v[-1,:] - 15*v[-2,:] - 4*v[-3,:] + 14*v[-4,:] - 6*v[-5,:] + v[-6,:]))/(12*dksi2_) \
        + np.multiply((-3*v[-1,:] - 10*v[-2,:] + 18*v[-3,:] - 6*v[-4,:] + v[-5,:]), 
                      (-3*A22[-1,:] - 10*A22[-2,:] + 18*A22[-3,:] - 6*A22[-4,:] + A22[-5,:]))/(144*dksi2_)
        
    # next-to-next-to boundary points
    v_xx[:,2] = np.multiply(A11[:,2], (-v[:,0] + 16*v[:,1] - 30*v[:,2] + 16*v[:,3] - v[:,4]))/(12*dksi2_) \
        + np.multiply((v[:,0] - 8*v[:,1] + 8*v[:,3] - v[:,4]), (A11[:,0] - 8*A11[:,1] + 8*A11[:,3] - A11[:,4]))/(144*dksi2_)
        
    v_yy[2,:] = np.multiply(A22[2,:], (-v[0,:] + 16*v[1,:] - 30*v[2,:] + 16*v[3,:] - v[4,:]))/(12*dksi2_) \
        + np.multiply((v[0,:] - 8*v[1,:] + 8*v[3,:] - v[4,:]), (A11[0,:] - 8*A11[1,:] + 8*A11[3,:] - A11[4,:]))/(144*dksi2_)
        
    v_xx[:,-3] = np.multiply(A11[:,-3], (-v[:,-1] + 16*v[:,-2] - 30*v[:,-3] + 16*v[:,-4] - v[:,-5]))/(12*dksi2_) \
        + np.multiply((v[:,-1] - 8*v[:,-2] + 8*v[:,-4] - v[:,-5]), 
                      (A11[:,-1] - 8*A11[:,-2] + 8*A11[:,-4] - A11[:,-5]))/(144*dksi2_)
    
    v_yy[-3,:] = np.multiply(A22[-3,:], (-v[-1,:] + 16*v[-2,:] - 30*v[-3,:] + 16*v[-4,:] - v[-5,:]))/(12*dksi2_) \
        + np.multiply((v[-1,:] - 8*v[-2,:] + 8*v[-4,:] - v[-5,:]), 
                      (A11[-1,:] - 8*A11[-2,:] + 8*A11[-4,:] - A11[-5,:]))/(144*dksi2_)
    
    # B.2 (A12*u_deta)_ksi, (A12*u_dksi)_eta
    temp = M.dksiCentre*np.multiply(A12, v_deta)
    temp[Ibdy.Boundary] = 0
    v_xx += temp
    
    temp = M.dksiCentre*np.multiply(A12, v_dksi)
    temp[Ibdy.Boundary] = 0
    v_xx += temp
    
    return np.divide(np.reshape(v_xx, NN_), J), np.divide(np.reshape(v_yy, NN_), J)
    
        
    
def compute_monitor():
    """
    computes the monitor functino Mon
    for epsilon == 0
        mon = 1/(1+u)^6
    for epsilon > 0
        for p = 1
            mon = 1 + (u_x)^2 + (u_y)^2
        for p = 2
            mon = |u_xx + u_yy|^2
    """
    global monitor
    if epsilon_ == 0:
        monitor =  1/(1+u.val)**6
    else:
        if p_ == 1:
            monitor = 1 + u.dx**2 + u.dy**2
        else:
            monitor = np.sqrt(np.abs(u.d2x + u.d2y))  

def solve_PMA():
    """
    solve for dQdt
    """
    global J
    compute_Q_spatial_ders()
    J = np.multiply(Q.d2ksi, Q.d2eta) - np.multiply(Q.dksideta, Q.dksideta)
    compute_u_spatial_ders()
    compute_monitor()
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
