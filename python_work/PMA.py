# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:18:42 2020

@author: savva
"""
import numpy as np
from scipy.integrate import odeint

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
Q = None # mesh potential
Ibdy = lambda:0 # information on indices

def main():
    # initialise Q, Ibdy
    Q = np.reshape(0.5*ksiksi**2 + 0.5*etaeta**2, NN_)
    Ibdy = make_Ibdy()

def make_Ibdy():
    global Ibdy
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
    

if (__name__ == "__main__"):
    main()