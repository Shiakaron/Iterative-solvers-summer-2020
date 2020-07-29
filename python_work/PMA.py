# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:18:42 2020

@author: savva
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron, csc_matrix
from scipy.fft import dct, idct
# from scipy.linalg import matrix_power
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LightSource

np.set_printoptions(edgeitems=6, suppress=True)
i = 0

#GLOBAL variables
N_ = 50 # grid points
NN_ = N_*N_ # total number of points
p_ = 2 # set as 1 or 2
m_ = 3 # set as 3 (Van der Walls), or 4 (Casimir)
alpha_ = 0.1 # controls the mesh adaption speed 
gamma_ = 0.1 # controls the extent of smoothing
epsilon_ = 0
beta_ = 0.15
smoothing_iters_ = 4 # number of smoothing iterations per time step 
lambd_ = 10 # quantifies the relative importance of electrostatic and elastic forces in the system
endl_, endr_ = -1, 1
d_ = endr_ - endl_ # domain size
dksi_ = d_/(N_-1) # deta_ = dksi_
dksi2_ = dksi_*dksi_
Tf = 0.04 # solver should terminate before touchdown

#GLOBAL vectors, matrices
ksi = np.linspace(endl_, endr_, N_)
ksiksi, etaeta = np.meshgrid(ksi, ksi) # square grid
Ibdy = lambda:0 # information on indices (boundary, interior, corners)
M = lambda:0 # derivative matrices
Q = lambda:0 # mesh potential and all its derivatives
U = lambda:0 # solution and its derivatives
J = None # Hessian (Jacobian) of Q

#plotting in solver
fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')
ax.view_init(elev=30, azim=-120)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
ax.set_zlim3d(-1,.2)  
ls = LightSource(azdeg=0, altdeg=65)
surf = ax.plot_surface(ksiksi, etaeta, np.zeros((N_,N_)), cmap='viridis')

def main():
    global Q, Ibdy, M
    # initialise Q, Ibdy, M, u
    Q.val = np.reshape(0.5*ksiksi**2 + 0.5*etaeta**2, NN_) # mesh potential
    make_Ibdy()
    make_M()
    # U.val = np.zeros(NN_, dtype=float)
    U.val = -0.01*np.exp(-35*(ksiksi**2+etaeta**2)).reshape(NN_)
    
    # termination event
    touchdown.terminal=True
    touchdown.direction=-1
    # ode solver
    sol = solve_ivp(ode_coupled_systems,  (0,Tf), np.concatenate((U.val, Q.val)), method="BDF")
    # ode solve without mesh adaptation
    # sol = solve_ivp(ode_coupled_systems,  (0,Tf), U.val, events=touchdown)
    print(sol.message)
    
    plot = False
    # for plotting
    if plot == True:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        ax.view_init(elev=30, azim=-120)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
        ax.set_zlim3d(-1,.2)  
        ls = LightSource(azdeg=0, altdeg=65)
        surf = ax.plot_surface(ksiksi, etaeta, np.zeros((N_,N_)), cmap='viridis')
        plt.show()
        print("plotting a total of ", len(sol.t), " frames")
        for i in range(len(sol.t)):
            # Update plots
            surf.remove() 
            surf = ax.plot_surface(ksiksi, etaeta, (sol.y[:NN_,i]).reshape(N_,N_), \
                                    color=(0,0.8,1),rstride=1, cstride=1, linewidth=0, \
                                    antialiased=False,alpha=0.5)
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.suptitle("frame: "+str(i)+", time: "+str(sol.t[i]))
    
    
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
    
    # Smoothing Matrix
    # off1 = np.ones(NN_-1)
    # off1[(N_-1)::N_] = 0 
    # off2 = np.ones(NN_-3)
    # off2[0::N_] = 0 
    # M.Sm = diags([off1[:-N_],2,off2,2*off1,4,2*off1,off2,2,off1[:-N_]], \
    #              [-N_-1,-N_,-N_+1,-1,0,1,N_-1,N_,N_+1], shape=(NN_, NN_))/16
    
    # for discreet cosine transform
    temp = (2*np.cos(np.pi*np.arange(0,N_)/(N_-1))-2).reshape((N_,1))*np.ones(N_) + \
    np.ones((N_,1))*(2*np.cos(np.pi*np.arange(0,N_)/(N_-1))-2)
    M.Leig = temp/dksi2_

def compute_Q_spatial_ders():
    """
    compute spatial (ksi, eta) derivatives of the mesh potetial
    """
    # 1st derivatives
    Q.dksi = M.dksiCentre.dot(Q.val)
    Q.deta = M.detaCentre.dot(Q.val)
    # 2nd derivatives
    extra = 25/(6*dksi_)
    temp = np.zeros(NN_); temp[Ibdy.Left] = extra; temp[Ibdy.Right] = extra
    Q.d2ksi = M.d2ksi.dot(Q.val) + temp
    temp = np.zeros(NN_); temp[Ibdy.Top] = extra; temp[Ibdy.Bottom] = extra
    Q.d2eta = M.d2eta.dot(Q.val) + temp
    Q.dksideta = M.dksideta.dot(Q.val); Q.dksideta[Ibdy.Boundary] = 0

def compute_u_spatial_ders():
    """
    compute spatial (x, y) derivatives of the solution
    """
    # 1st derivatives (ksi, eta) - not saved in U struct
    U_dksi = M.dksiCentre.dot(U.val)
    U_deta = M.detaCentre.dot(U.val)
    # 1st derivatives (x, y)
    U.dx = np.divide(np.multiply(Q.d2eta,U_dksi) - np.multiply(Q.dksideta,U_deta), J)
    U.dy = np.divide(- np.multiply(Q.dksideta,U_dksi) + np.multiply(Q.d2ksi,U_deta), J)
    # 2nd derivatives (xx, yy) - laplacian operator
    U.xx, U.yy = Laplace_operator(np.reshape(U.val,(N_,N_)), U_dksi, U_deta)
    
def Laplace_operator(v, v_dksi, v_deta):
    """    
    L(v) = v_xx + v_yy = J^-1 * div_ksi { J^-1 * A * grad_ksi (v) }
    more consicely:
    v_xx = J^-1 * [ ( A11 * v_dksi )_ksi + ( A12 * v_deta)_ksi ]
    v_yy = J^-1 * [ ( A12 * v_dksi )_eta + ( A22 * v_deta)_eta ]
    where appendix B shows the discretisation of the bracketed terms.
       
    The argument v needs to be an N_xN_ array
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
    temp = M.dksiCentre.dot(np.multiply(A12, v_deta))
    temp[Ibdy.Boundary] = 0
    v_xx = np.reshape(v_xx, NN_)
    v_xx += temp
    
    temp = M.dksiCentre.dot(np.multiply(A12, v_dksi))
    temp[Ibdy.Boundary] = 0
    v_yy = np.reshape(v_yy, NN_)
    v_yy += temp
    
    return np.divide(v_xx, J), np.divide(v_yy, J)     
    
def compute_and_smooth_monitor():
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
    # initialise arrays
    temp = np.zeros((N_, N_), dtype=float)
    mon = np.zeros((N_, N_), dtype=float)
    
    # compute monitor function
    if epsilon_ == 0:
        temp =  (1/(1+U.val)**6).reshape((N_,N_))
    else:
        if p_ == 1:
            temp = (1 + U.dx**2 + U.dy**2).reshape((N_,N_))
        else:
            temp = np.sqrt(np.abs(U.d2x + U.d2y)).reshape((N_,N_))
    
    # smoothing           
    # fourth-order filter
    for i in range(smoothing_iters_):
        # interior points
        mon[1:-1,1:-1] = temp[1:-1,1:-1] + (temp[:-2,1:-1] + temp[2:,1:-1] + temp[1:-1,:-2] + temp[1:-1,2:])/8 \
            + (temp[:-2,:-2] + temp[:-2,2:] + temp[2:,:-2] + temp[2:,2:])/16
        # boundary but no corners
        mon[1:-1,N_-1] = (4*temp[1:-1,N_-1] + 2*temp[:-2,N_-1] +2*temp[2:,N_-1] + 2*temp[1:-1,N_-2] + temp[2:,N_-2] + temp[:-2,N_-2])/12
        mon[1:-1,0] = (4*temp[1:-1,0] + 2*temp[:-2,0] +2*temp[2:,0] + 2*temp[1:-1,1] + temp[2:,1] + temp[:-2,1])/12
        mon[N_-1,1:-1] = (4*temp[N_-1,1:-1] + 2*temp[N_-1,:-2] +2*temp[N_-1,2:] + 2*temp[N_-2,1:-1] + temp[N_-2,2:] + temp[N_-2,:-2])/12
        mon[0,1:-1] = (4*temp[0,1:-1] + 2*temp[0,:-2] +2*temp[0,2:] + 2*temp[1,1:-1] + temp[1,2:] + temp[1,:-2])/12
        # corners
        mon[0,0] = (4*temp[0,0] + 2*temp[0,1] + 2*temp[1,0] + temp[1,1])/9
        mon[0,N_-1] = (4*temp[0,N_-1] + 2*temp[0,N_-2] + 2*temp[1,N_-1] + temp[1,N_-2])/9
        mon[N_-1,0] = (4*temp[N_-1,0] + 2*temp[N_-1,1] + 2*temp[N_-2,0] + temp[N_-2,1])/9
        mon[N_-1,N_-1] = (4*temp[N_-1,N_-1] + 2*temp[N_-1,N_-2] + 2*temp[N_-2,N_-1] + temp[N_-2,N_-2])/9
        # update temp        
        temp = mon.copy()
    mon = np.reshape(mon, NN_)
    # Mackenzie regularisation    
    mon_integral = np.sum(mon*np.abs(J))*dksi2_
    mon += mon_integral
    return mon

def solve_PMA():
    """
    solve for dQdt = L.fancy^-1 * (|J|*M)^0/5
    L.fancy^-1 is the inverse of the operator L.fancy = aplha*(Identity - gamma*Lap_ξ)
    which is solved using discreet cosine transform
    """
    monitor = compute_and_smooth_monitor()
    q_rhs = np.sqrt(np.multiply(monitor, np.abs(J)))/alpha_
    temp = dct(q_rhs.reshape(N_,N_), norm="ortho")
    dQdt = idct(np.divide(temp,(1-gamma_*M.Leig)))
    return dQdt.reshape(NN_)

def compute_rhs_pde(Qt):
    """
    solve for dudt = -(-Lap)^p(u) - lambda/(1+u)^2 + lambda*epsilon^(m-2)/(1+u)^m + Langrangian_term
    where the Langrangian_term = Grad_x{u} dot Grad_ksi{Qt}.
    """
    dudt = - lambd_/((1+U.val)**2) + lambd_*(epsilon_**(m_-2))/((1+U.val)**m_)
    dudt += langrangian_term(Qt)
    if p_ == 1:
        dudt += beta_*beta_*(U.xx + U.yy)
    else: # p == 2
        v = U.xx + U.yy
        v_dksi = M.dksiCentre*v
        v_deta = M.detaCentre*v
        v_xx, v_yy = Laplace_operator(v.reshape(N_,N_), v_dksi, v_deta)
        dudt -= beta_*beta_*(v_xx + v_yy)
    dudt[Ibdy.Boundary] = 0
    return dudt

def langrangian_term(Qt):
    """
    See appendix C
    Langrangian_term = Grad_x{u} dot Grad_ksi{Q_t}.
    Grad_ksi{Qt} = (a, b) = (Qt_dksi, Qt_deta)
    """
    a = M.dksiCentre.dot(Qt)
    b = M.detaCentre.dot(Qt)
    ret = np.zeros(NN_, dtype=float)
    U_ksi_forw = M.dksiForw.dot(U.val)
    U_ksi_back = M.dksiBack.dot(U.val)
    U_eta_forw = M.detaForw.dot(U.val)
    U_eta_back = M.detaBack.dot(U.val)
    # upwinding in x direction
    ret += np.minimum(a,0)*(np.maximum(b,0)*np.divide((Q.d2eta*U_ksi_back - Q.dksideta*U_eta_forw), J) +
                            np.minimum(b,0)*np.divide((Q.d2eta*U_ksi_back - Q.dksideta*U_eta_back), J)) \
         + np.maximum(a,0)*(np.maximum(b,0)*np.divide((Q.d2eta*U_ksi_forw - Q.dksideta*U_eta_forw), J) +
                            np.minimum(b,0)*np.divide((Q.d2eta*U_ksi_forw - Q.dksideta*U_eta_back), J))
    # upwinding in y direction
    ret += np.minimum(a,0)*(np.maximum(b,0)*np.divide((-Q.dksideta*U_ksi_back + Q.d2ksi*U_eta_forw), J) +
                            np.minimum(b,0)*np.divide((-Q.dksideta*U_ksi_back + Q.d2ksi*U_eta_back), J)) \
         + np.maximum(a,0)*(np.maximum(b,0)*np.divide((-Q.dksideta*U_ksi_forw + Q.d2ksi*U_eta_forw), J) +
                            np.minimum(b,0)*np.divide((-Q.dksideta*U_ksi_forw + Q.d2ksi*U_eta_back), J))
    return ret

def touchdown(t, y):
    return min(y) + 0.5

def ode_coupled_systems(t, y):
    """
    """
    global J 
    # assign U, Q, (g?)
    U.val = y[:NN_]
    Q.val = y[NN_:] 
    
    # compute derivatives
    compute_Q_spatial_ders()
    J = np.multiply(Q.d2ksi, Q.d2eta) - np.multiply(Q.dksideta, Q.dksideta)
    compute_u_spatial_ders()
    
    # solve PMA for dQdt
    dQdt = solve_PMA()
    # dQdt = np.zeros(NN_)
    
    # solve rhs of actual problem for dudt      
    dudt = compute_rhs_pde(dQdt)
    
    # iter
    global i, surf
    i += 1
    print("iteration : ", i, "{:e}".format(t))
    
    # Update plots
    if i%1000 == 0:
        surf.remove() 
        surf = ax.plot_surface(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), U.val.reshape(N_,N_), \
                                color=(0,0.8,1),rstride=1, cstride=1, linewidth=0, \
                                antialiased=False,alpha=0.5)
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.suptitle("frame: "+str(i/1000))
    
    return np.concatenate((dudt, dQdt))
    # return dudt

  
def compute_g(u):
    if epsilon_ == 0:
        return min((1+u)**3)
    else:
        return 1
    
        
        
if (__name__ == "__main__"):
    main()
