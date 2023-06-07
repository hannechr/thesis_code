import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))

import numpy as np
from problem_SWE_m import problem_SWE
from solution_m import solution
from parareal_m import parareal
from intRK4_m import intRK4
from intBWE_m import intBWE
from intNN_SWE_m import intNN_SWE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

np.random.seed(125298133)

half_figs = (5.6,2.2)
fullfull_figs = (7, 8)
ms = 5
fs_legend = 7

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'savefig.pad_inches': 0.01
})


# Spatial grid
m = 16     # Number of grid points in space
L = 30000      # Width of spatial domain

# Wavenumber "grid"
xi = np.fft.fftfreq(m)*m*2*np.pi/L

# Physical variables
h_avg = 1000        # Average fluid heigth
h_ampl = 100         # Order of wave amplitudes
nu = 0              # Hyperviscosity coefficient
ord_hyp = 4         # Order of the hyperviscosity
theta = 0.01        # Angle at which simulation takes place
g = 9.81            # Gravitation
w_earth = 7.29e-5   # Earth rotational speed
f = 2*w_earth*np.sin(theta) # Coriolis effect

# define spatial operator matrix for linear part
D = np.array(([-nu*xi**(ord_hyp), f*np.ones_like(xi), -g*xi*1j],
              [ -f*np.ones_like(xi), -nu*xi**(ord_hyp), 0*np.ones_like(xi)],
              [-h_avg*xi*1j, 0*np.ones_like(xi), -nu*xi**(ord_hyp)]))  # shape nvar x nvar x nspat

# define nonlinear part: u_t = -u*u_x // v_t = - u*v_x // h'_t = -(u*h')_x
# output = nsets x nvar x nspat
def nlin(what):

    def convolve(ahat,bhat): # shape input: nsets x nspat
        nspat = np.shape(ahat)[1]
        padding = ((0,0),(nspat//2, nspat//2))
        ahat_ext = np.pad(np.fft.fftshift(ahat, axes=1), padding)
        bhat_ext = np.pad(np.fft.fftshift(bhat, axes=1), padding)
        a = 4*np.fft.ifft(ahat_ext, axis=1)
        b = 4*np.fft.ifft(bhat_ext, axis=1)

        # perform operation
        prod_ext = a * b

        # return to spectral domain
        prodhat_ext = np.fft.fft(prod_ext, axis=1)

        # crop high frequencies
        prodhat = np.fft.ifftshift(1/4*np.fft.fftshift(prodhat_ext, axes=1)[:, nspat//2:3*nspat//2], axes=1)
        return prodhat
    
    uhat = what[:,0,:]
    vhat = what[:,1,:]
    hhat = what[:,2,:]
    ddx = 1j*xi
    return - np.concatenate((convolve(ddx*uhat, uhat)[:,np.newaxis,:], \
                             convolve(ddx*vhat, uhat)[:,np.newaxis,:], \
                             ddx*convolve(uhat, hhat)[:,np.newaxis,:]), axis=1)

problem = problem_SWE(D, nlin, h_avg, h_ampl, L, nbneurons = 128)


tstart = 0
tend = 160

nfine = 16*10
ncoarse = 16*4

coarse = intNN_SWE
fine = intRK4

U0 = problem.get_rand_init(1)
uhat0 = np.fft.fft(U0, axis=2)


intgcoarse = coarse(tstart, tend, ncoarse, problem)
intgfine = fine(tstart, tend, nfine, problem)

start = time.time()
ycoarse = intgcoarse.run_last(solution(problem, uhat0))[0,2,:]
end = time.time()
print(end - start)

start = time.time()
yfine = intgfine.run_last(solution(problem, uhat0))[0,2,:]
end = time.time()
print(end - start)


