# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 09:29:12 2020

@author: savva
"""
import numpy as np
from scipy.sparse import diags, kron, csc_matrix
from scipy.fft import dct, idct
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import csv
import os

np.set_printoptions(edgeitems=6, suppress=True)

#GLOBAL parameters
R_ = 1 # radius of droplet
a_ = 100
epsilon_ = 1e-5 # thin liquid layer height
V_, Vf_ = 0, 1 # volume of droplet (starts from 0 and stops at 1)
Vsteps_ = 1000 # number of steps for droplet initialisation

#GLOBAL simulation variables
N_ = 51 # grid points
NN_ = N_*N_ # total number of points
smoothing_iters_ = 4 # number of smoothing iterations per time step 
endl_, endr_ = -2.5, 2.5
d_ = endr_ - endl_ # domain size
dksi_ = d_/(N_-1) # deta_ = dksi_
dksi2_ = dksi_*dksi_

#PMA variables
alpha_ = 0.01 # controls the mesh adaption speed 
gamma_ = 0.1 # controls the extent of smoothing
C_ = .25 # Mackenzie normalisation constant
dtmesh_ = 5e-7

#PDE variables
dtR_ = 5e-2

#GLOBAL vectors/matrices/terms
ksi = np.linspace(endl_, endr_, N_)
ksiksi, etaeta = np.meshgrid(ksi, ksi) # square grid
Ibdy = lambda:0 # information on indices (boundary, interior, corners)
M = lambda:0 # derivative matrices
Q = lambda:0 # mesh potential and all its derivatives
U = lambda:0 # solution and its derivatives
J = None # Hessian (Jacobian) of Q

#writing to or reading from file?
fromfile = True
tofile = True

# for plotting
plot3d_bool = True
fig = plt.figure(figsize=(16,8))
aaa = np.linspace(0,2*np.pi)
if plot3d_bool:
    # solution and mesh
    ax = fig.add_subplot(121, projection='3d') 
    ax.view_init(elev=30, azim=-160)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
    ax.set_zlim3d(-3,4.2)  
    ax.grid(False)
    ls = LightSource(azdeg=50, altdeg=65)
    surf = ax.plot_surface(ksiksi, etaeta, np.zeros((N_,N_)))
    mesh = ax.plot_wireframe(ksiksi, etaeta, np.zeros((N_,N_)))
    # mesh
    ax2 = fig.add_subplot(122, projection='3d') 
    ax2.view_init(elev=90, azim=0)
    ax2.set_xlabel('x'); ax2.set_ylabel('y');
    ax2.w_zaxis.line.set_lw(0.)
    ax2.set_zticks([])
    ax2.grid(False)
    radiusline = ax2.plot3D(R_*np.cos(aaa),R_*np.sin(aaa),0*aaa,'r',linewidth=0.2)
    ls2 = LightSource(azdeg=50, altdeg=65)
    mesh2 = ax.plot_wireframe(ksiksi, etaeta, np.zeros((N_,N_)))
    # 
    plt.subplots_adjust(left= 0, bottom=0, right=1, top=1, wspace = 0)

def main():
    """
    """
    global Q, Ibdy, M, J, CN_term, surf, mesh, mesh2, V_, alpha_, gamma_, dtmesh_
    # initialise Q, Ibdy, M, U and droplet
    Q.val = np.reshape(0.5*ksiksi**2 + 0.5*etaeta**2, NN_) # mesh potential
    make_Ibdy()
    make_M()
    U.new = np.full(NN_, epsilon_)
    
    # initialise droplet
    if fromfile:
        read_from_file("initdrop_"+str(R_)+"_"+str(N_)+"_"+str(a_)+"_"+str(alpha_)+"_"+str(gamma_)+"_"+str(C_)+".txt")
        V_ = Vf_
        # plot
        compute_Q_spatial_ders()
        surf.remove() 
        surf = ax.plot_surface(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), U.val.reshape(N_,N_), \
                                    cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, \
                                    antialiased=False,alpha=0.5)
        mesh.remove() 
        mesh = ax.plot_wireframe(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), np.full((N_,N_),-3), linewidth=0.2)
        mesh2.remove()
        mesh2 = ax2.plot_wireframe(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), np.zeros((N_,N_)), linewidth=0.2, rcount=N_, ccount=N_)
        fig.canvas.draw()
        fig.canvas.flush_events() 
    else:
        initialise_droplet(dtmesh_, 1)
        
    # check PMA steady state
    check_mesh(dtmesh_, 8e-5)     
    
    # evolve droplet radius explicitely and update mesh
    alpha_ = 0.1
    dtmesh_ = 1e-6
    evolve_R_explicit(20, 2, 1e-2)

def initialise_droplet(dtmesh, loops):
    print("initialising droplet")
    global J, V_, surf, mesh, mesh2
    for i in range(1,Vsteps_+1):
        U.val = U.new.copy()
        #compute derivatives
        compute_Q_spatial_ders()
        J = Q.d2ksi*Q.d2eta - Q.dksideta**2
        compute_u_spatial_ders()
        #update solution
        V_ = Vf_*i/Vsteps_
        U.new = compute_U()
        print(V_, U.new.max())
        #solve PMA and update mesh 
        loop_pma(dtmesh, loops)
        # plot every once in a while
        if plot3d_bool:
            # solution
            surf.remove() 
            surf = ax.plot_surface(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), U.new.reshape(N_,N_), \
                                    cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, \
                                    antialiased=False,alpha=0.5)
            mesh.remove()
            mesh = ax.plot_wireframe(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), np.full((N_,N_),-3), linewidth=0.2)
            # mesh
            mesh2.remove()
            mesh2 = ax2.plot_wireframe(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), np.zeros((N_,N_)), linewidth=0.2, rcount=N_, ccount=N_)
            fig.canvas.draw()
            fig.canvas.flush_events() 
    U.val = U.new.copy()
    print("initialisation complete")
    
    # write values to a file for quick access??
    if tofile:
        write_to_file("initdrop_"+str(R_)+"_"+str(N_)+"_"+str(a_)+"_"+str(alpha_)+"_"+str(gamma_)+"_"+str(C_)+".txt")

def check_mesh(dt, atol):
    # to check steady state of the mesh
    global J, V_, surf, mesh, mesh2
    iters = 1000
    # start evolving 
    compute_Q_spatial_ders()
    J = Q.d2ksi*Q.d2eta - Q.dksideta**2
    compute_u_spatial_ders()
    for i in range(iters):
        #solve PMA and find differences between updated mesh before updating 
        solve_PMA()
        Qnew = Q.val.copy() + dt*Q.dt
        # 1st derivatives
        Qdksi = M.dksiCentre.dot(Qnew); Qdksi[Ibdy.Left] = endl_; Qdksi[Ibdy.Right] = endr_;
        Qdeta = M.detaCentre.dot(Qnew); Qdeta[Ibdy.Bottom] = endl_; Qdeta[Ibdy.Top] = endr_;
        diff_ksi = Qdksi - Q.dksi; diff_eta = Qdeta - Q.deta
        diff_squared = np.sqrt(diff_ksi**2 + diff_eta**2)
        print((i+1), " / ", iters, ": ", diff_squared.max())
        compute_Q_spatial_ders()
        J = Q.d2ksi*Q.d2eta - Q.dksideta**2
        Q.val = Qnew.copy()
        # update solution
        U.val = compute_U()
        # plot every once in a while
        if plot3d_bool:
            # solution
            surf.remove() 
            surf = ax.plot_surface(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), U.val.reshape(N_,N_), \
                                    cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, \
                                    antialiased=False,alpha=0.5)
            mesh.remove()
            mesh = ax.plot_wireframe(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), np.full((N_,N_),-3), linewidth=0.2)
            #mesh
            mesh2.remove()
            mesh2 = ax2.plot_wireframe(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), np.zeros((N_,N_)), linewidth=0.2, \
                                       rcount=N_, ccount=N_)
            fig.canvas.draw()
            fig.canvas.flush_events() 
        if diff_squared.max() < atol:
            print("PMA steady state achieved with atol = "+str(atol))
            break

def evolve_R_explicit(pmaloops, Rfinal, tol):
    global R_, surf, mesh, mesh2, J, radiusline
    time = 0
    i = 0
    U.val = U.new.copy()
    while (abs(Rfinal - R_) > tol):
        print(time, R_, U.val.max())
        #compute derivatives
        compute_Q_spatial_ders()
        J = Q.d2ksi*Q.d2eta - Q.dksideta**2
        compute_u_spatial_ders()
        # timestep
        dt = dtR_*(R_**2)
        # new radius
        R_ += dt*Rdot()
        U.val = compute_U()
        #solve PMA and update mesh 
        loop_pma(dtmesh_, pmaloops)
        # iteration
        i += 1
        time += dt
        # plot
        if plot3d_bool:
            # solution
            surf.remove() 
            surf = ax.plot_surface(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), U.val.reshape(N_,N_), \
                                    cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, \
                                    antialiased=False,alpha=0.5)
            mesh.remove()
            mesh = ax.plot_wireframe(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), np.full((N_,N_),-3), linewidth=0.2)
            # mesh
            ax2.clear()
            ax2.w_zaxis.line.set_lw(0.)
            ax2.set_zticks([])
            ax2.grid(False)
            mesh2 = ax2.plot_wireframe(Q.dksi.reshape(N_,N_), Q.deta.reshape(N_,N_), np.zeros((N_,N_)), linewidth=0.2, rcount=N_, ccount=N_)
            # update circle
            radiusline = ax2.plot3D(R_*np.cos(aaa),R_*np.sin(aaa),0*aaa,'r',linewidth=0.1, alpha=0.7)
            fig.canvas.draw()
            fig.canvas.flush_events() 
            
            

def compute_U():
    return epsilon_ + (1-epsilon_)*H(G(np.sqrt(Q.dksi*Q.dksi+Q.deta*Q.deta)))

def H(psi):
    ret = 4*V_*(1-psi*psi/(R_*R_))/(R_*R_)
    return np.where(ret > 0, ret, 0)

def G(x):
    return R_ + np.log((1+np.exp(-2*a_*(x+R_)))/(1+np.exp(-2*a_*(x-R_))))/(2*a_)
     
def Rdot():
    return (8*V_/R_**3 - 1)/(3*np.log(1/epsilon_)) 

def write_to_file(filename):
    print("writing in file")
    file = open(filename, "w") 
    for i in range(NN_):
        # U.val Q.val 
        file.write(str(U.val[i]) + " " + str(Q.val[i]) + "\n")
    file.close()
    
def read_from_file(filename):
    print("reading intialised state from file")
    a = []
    b = []
    p_file = os.getcwd() + "\\" + filename
    with open(p_file) as csvfile:
        lines = csv.reader(csvfile, delimiter=' ')
        for row in lines:
            a.append(float(row[0]))
            b.append(float(row[1]))
    U.new = np.array(a)
    U.val = U.new.copy()
    Q.val = np.array(b)

def solve_PMA():
    """
    solve for dQdt = L.fancy^-1 * (|J|*M)^0/5
    L.fancy^-1 is the inverse of the operator L.fancy = aplha*(Identity - gamma*Lap_ξ)
    which is solved using discreet cosine transform
    """
    monitor = compute_and_smooth_monitor()
    q_rhs = np.sqrt(np.multiply(monitor, np.abs(J)))/alpha_
    temp = dct(dct(q_rhs.reshape(N_,N_).T, norm="ortho").T, norm="ortho")
    dQdt = idct(idct(np.divide(temp,(1-gamma_*M.Leig)).T, norm="ortho").T, norm="ortho") 
    Q.dt = dQdt.reshape(NN_)
    
def loop_pma(dt, loops):
    global J
    solve_PMA()
    Q.val += dt*Q.dt
    for i in range(1,loops):
        compute_Q_spatial_ders()
        J = Q.d2ksi*Q.d2eta - Q.dksideta**2
        compute_u_spatial_ders()
        solve_PMA()
        Q.val += dt*Q.dt

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
    v_xx = np.zeros((N_, N_), dtype=float); v_yy = np.zeros((N_, N_), dtype=float)
    
    # B.1 : (A11*u_dksi)_ksi, (A22*u_deta)_eta
    # interior points  (checked 31/7 - all good)
    v_xx[:,3:-3] = (4*np.multiply(A11[:,2:-4], (v[:,:-6] - 8*v[:,1:-5] + 8*v[:,3:-3] - v[:,4:-2])) 
                  - np.multiply((- A11[:,1:-5] + 9*A11[:,2:-4] + 9*A11[:,3:-3] - A11[:,4:-2]), 
                               (v[:,1:-5] - 27*v[:,2:-4] + 27*v[:,3:-3] - v[:,4:-2])) 
                  + np.multiply((- A11[:,2:-4] + 9*A11[:,3:-3] + 9*A11[:,4:-2] - A11[:,5:-1]), 
                               (v[:,2:-4] - 27*v[:,3:-3] + 27*v[:,4:-2] - v[:,5:-1]))
                  - 4*np.multiply(A11[:,4:-2], (v[:,2:-4] - 8*v[:,3:-3] + 8*v[:,5:-1] - v[:,6:])))/(288*dksi2_)
    
    v_yy[3:-3,:] = (4*np.multiply(A22[2:-4,:], (v[:-6,:] - 8*v[1:-5,:] + 8*v[3:-3,:] - v[4:-2,:]))
                  - np.multiply((- A22[1:-5,:] + 9*A22[2:-4,:] + 9*A22[3:-3,:] - A22[4:-2,:]), 
                               (v[1:-5,:] - 27*v[2:-4,:] + 27*v[3:-3,:] - v[4:-2,:]))
                  + np.multiply((- A22[2:-4,:] + 9*A22[3:-3,:] + 9*A22[4:-2,:] - A22[5:-1,:]), 
                               (v[2:-4,:] - 27*v[3:-3,:] + 27*v[4:-2,:] - v[5:-1,:]))
                  - 4*np.multiply(A22[4:-2,:], (v[2:-4,:] - 8*v[3:-3,:] + 8*v[5:-1,:] - v[6:,:])))/(288*dksi2_)
    
    # next-to boundary points (checked 31/7 - reversed sign of components of expressions for [:,-2] and [-2,:], this shouldn't 
    # actually change anything.)
    v_xx[:,1] = np.multiply(A11[:,1], (10*v[:,0] - 15*v[:,1] - 4*v[:,2] + 14*v[:,3] - 6*v[:,4] + v[:,5]))/(12*dksi2_) \
                + np.multiply((-3*v[:,0] - 10*v[:,1] + 18*v[:,2] - 6*v[:,3] + v[:,4]), 
                              (-3*A11[:,0] - 10*A11[:,1] + 18*A11[:,2] - 6*A11[:,3] + A11[:,4]))/(144*dksi2_)
        
    v_yy[1,:] = np.multiply(A22[1,:], (10*v[0,:] - 15*v[1,:] - 4*v[2,:] + 14*v[3,:] - 6*v[4,:] + v[5,:]))/(12*dksi2_) \
                + np.multiply((-3*v[0,:] - 10*v[1,:] + 18*v[2,:] - 6*v[3,:] + v[4,:]), 
                              (-3*A22[0,:] - 10*A22[1,:] + 18*A22[2,:] - 6*A22[3,:] + A22[4,:]))/(144*dksi2_)
    
    v_xx[:,-2] = np.multiply(A11[:,-2], (10*v[:,-1] - 15*v[:,-2] - 4*v[:,-3] + 14*v[:,-4] - 6*v[:,-5] + v[:,-6]))/(12*dksi2_) \
                + np.multiply((3*v[:,-1] + 10*v[:,-2] - 18*v[:,-3] + 6*v[:,-4] - v[:,-5]), 
                              (3*A11[:,-1] + 10*A11[:,-2] - 18*A11[:,-3] + 6*A11[:,-4] - A11[:,-5]))/(144*dksi2_)
        
    v_yy[-2,:] = np.multiply(A22[-2,:], (10*v[-1,:] - 15*v[-2,:] - 4*v[-3,:] + 14*v[-4,:] - 6*v[-5,:] + v[-6,:]))/(12*dksi2_) \
                + np.multiply((3*v[-1,:] + 10*v[-2,:] - 18*v[-3,:] + 6*v[-4,:] - v[-5,:]), 
                              (3*A22[-1,:] + 10*A22[-2,:] - 18*A22[-3,:] + 6*A22[-4,:] - A22[-5,:]))/(144*dksi2_)
        
    # next-to-next-to boundary points (checked 31/7 - adjasted ordering for components of [:,-3] and [-3,:]
    # to ascending order, this shouldn't actually change anything. Also FIXED A11->A22 for the appropriate expressions.)
    v_xx[:,2] = np.multiply(A11[:,2], (- v[:,0] + 16*v[:,1] - 30*v[:,2] + 16*v[:,3] - v[:,4]))/(12*dksi2_) \
                + np.multiply((v[:,0] - 8*v[:,1] + 8*v[:,3] - v[:,4]), 
                              (A11[:,0] - 8*A11[:,1] + 8*A11[:,3] - A11[:,4]))/(144*dksi2_)
        
    v_yy[2,:] = np.multiply(A22[2,:], (- v[0,:] + 16*v[1,:] - 30*v[2,:] + 16*v[3,:] - v[4,:]))/(12*dksi2_) \
                + np.multiply((v[0,:] - 8*v[1,:] + 8*v[3,:] - v[4,:]), 
                              (A22[0,:] - 8*A22[1,:] + 8*A22[3,:] - A22[4,:]))/(144*dksi2_)
        
    v_xx[:,-3] = np.multiply(A11[:,-3], (- v[:,-1] + 16*v[:,-2] - 30*v[:,-3] + 16*v[:,-4] - v[:,-5]))/(12*dksi2_) \
                + np.multiply((v[:,-5] - 8*v[:,-4] + 8*v[:,-2] - v[:,-1]), 
                              (A11[:,-5] - 8*A11[:,-4] + 8*A11[:,-2] - A11[:,-1]))/(144*dksi2_)
    
    v_yy[-3,:] = np.multiply(A22[-3,:], (- v[-1,:] + 16*v[-2,:] - 30*v[-3,:] + 16*v[-4,:] - v[-5,:]))/(12*dksi2_) \
                + np.multiply((v[-5,:] - 8*v[-4,:] + 8*v[-2,:] - v[-1,:]), 
                              (A22[-5,:] - 8*A22[-4,:] + 8*A22[-2,:] - A22[-1,:]))/(144*dksi2_)
    
    # B.2 (A12*u_deta)_ksi, (A12*u_dksi)_eta (checked 31/7 - fixed major mistakes)
    temp = M.dksiCentre.dot(np.multiply(A12, v_deta))
    temp[Ibdy.Left] = 0; temp[Ibdy.Right] = 0
    v_xx = np.reshape(v_xx, NN_)
    v_xx += temp
    
    temp = M.detaCentre.dot(np.multiply(A12, v_dksi))
    temp[Ibdy.Top] = 0; temp[Ibdy.Bottom] = 0
    v_yy = np.reshape(v_yy, NN_)
    v_yy += temp
    
    return np.divide(v_xx, J), np.divide(v_yy, J)
    
def compute_Q_spatial_ders():
    """
    compute spatial (ksi, eta) derivatives of the mesh potetial
    """
    # 1st derivatives
    Q.dksi = M.dksiCentre.dot(Q.val); Q.dksi[Ibdy.Left] = endl_; Q.dksi[Ibdy.Right] = endr_;
    Q.deta = M.detaCentre.dot(Q.val); Q.deta[Ibdy.Bottom] = endl_; Q.deta[Ibdy.Top] = endr_;
    # 2nd derivatives
    extra = 25/(6*dksi_)*endr_
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
 
def compute_and_smooth_monitor():
    """
    computes the monitor functino Mon
    mon = |u_xx + u_yy|^2
    """
    # initialise arrays
    mon = np.zeros((N_, N_), dtype=float)
    temp = (np.abs(U.xx + U.yy)**2).reshape((N_,N_))
    
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
    mon += C_*mon_integral
    return mon    
    
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
    temp = diags([-3, 4,-1], [0,1,2], shape=(N_,N_), format="lil")
    temp[-1,-3:] = [1, -4, 3]
    temp[-2,-2:] = [-2, 2]
    temp - csc_matrix(temp/(2*dksi_))
    M.dksiForw = kron(eye, temp) # df/dksi forward difference
    M.detaForw = kron(temp, eye) # df/deta forward difference
    
    #backward
    temp = diags([1,-4,3], [-2,-1,0], shape=(N_,N_), format="lil")
    temp[0,:3] = [-3, 4, -1]
    temp[1,:2] = [-2, 2]
    temp - csc_matrix(temp/(2*dksi_))
    M.dksiBack = kron(eye, temp) # df/dksi backward difference
    M.detaBack = kron(temp, eye) # df/deta backward difference
    
    # for discreet cosine transform
    temp = (2*np.cos(np.pi*np.arange(0,N_)/(N_-1))-2).reshape((N_,1))*np.ones(N_) + \
    np.ones((N_,1))*(2*np.cos(np.pi*np.arange(0,N_)/(N_-1))-2)
    M.Leig = temp/dksi2_

if (__name__ == "__main__"):
    main()
