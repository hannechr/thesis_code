import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))

import numpy as np
from problem_linadv_m import problem_linadv
from solution_m import solution
from parareal_m import parareal
from intRK4_m import intRK4
from intBWE_m import intBWE
from intExact_linadv_m import intExact_linadv
from intNN_linadv_m import intNN_linadv
import matplotlib.pyplot as plt
import matplotlib
from pylab import rcParams
from subprocess import call

half_figs = (5.5,2.0)
fullfull_figs = (7, 8)
ms = 5
fs_legend = 7

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# problem parameters
uadv = 1.0/4    # advection speed
nu = 0.0        # diffusivity parameter
nvar = 1        # number of variables

# Spatial grid
m = 16    # Number of grid points in space
L = 1.0   # Width of spatial domain
xi = np.fft.fftfreq(m)*m*2*np.pi/L

# define spatial operator matrix
D = -1j*xi*uadv - nu*xi**2
D = np.reshape(D, (1,m))
nbneurons = 32

problem = problem_linadv(D, uadv, nu, L, nbneurons)

# initial condition
u0    = problem.get_rand_init(1, L=L)
uhat0 = np.fft.fft(u0, axis=2)
initsol = solution(problem, uhat0)

# temporal specifications
tstart = 0
tend = 16


## Backward Euler - vary nb steps

# parareal parameters
tol = 1e-12              # tolerance
nslices = int(tend)     # time slices
coarse = intBWE         # coarse integrator class
fine = intExact_linadv  # fine integrator class
itmax = nslices         # maximal nb of iterations

dtlist =  np.array((0.0025, 0.010, 0.025, 0.100, 0.2500, 1))
ncoarse_all = (1./dtlist).astype(int)
res = np.zeros((nvar, itmax, len(ncoarse_all)))

fig1 = plt.figure(figsize=half_figs)
plt.subplot(1,2,1)

for i in range(len(ncoarse_all)):
    ncoarse = int(ncoarse_all[i])
    nfine = ncoarse
    para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=itmax, u0=solution(problem, uhat0))
    it, res[:,:,i] = para.run()
    plt.semilogy(range(1,itmax),res[0,:-1,i], label=f"$\Delta t = {dtlist[i]}$")

plt.ylim(top=1e2)
plt.ylim(bottom=1e-10)
plt.legend(fontsize=fs_legend)
# plt.title("bwe, vary coarse time step")
plt.xlabel("$k$")
plt.ylabel("$E^k$")


# problem parameters
uadv = 1.0/4    # advection speed
nu = 0.01        # diffusivity parameter
nvar = 1        # number of variables

# Spatial grid
m = 16    # Number of grid points in space
L = 1.0   # Width of spatial domain
xi = np.fft.fftfreq(m)*m*2*np.pi/L

# define spatial operator matrix
D = -1j*xi*uadv - nu*xi**2
D = np.reshape(D, (1,m))
nbneurons = 32

problem = problem_linadv(D, uadv, nu, L, nbneurons)

# initial condition
u0    = problem.get_rand_init(1, L=L)
uhat0 = np.fft.fft(u0, axis=2)
initsol = solution(problem, uhat0)

# temporal specifications
tstart = 0
tend = 16


## Backward Euler - vary nb steps

# parareal parameters
tol = 1e-12              # tolerance
nslices = int(tend)     # time slices
coarse = intBWE         # coarse integrator class
fine = intExact_linadv  # fine integrator class
itmax = nslices         # maximal nb of iterations

dtlist =  np.array((0.0025, 0.010, 0.025, 0.100, 0.2500, 1))
ncoarse_all = (1./dtlist).astype(int)
res = np.zeros((nvar, itmax, len(ncoarse_all)))
plt.subplot(1,2,2)
for i in range(len(ncoarse_all)):
    ncoarse = int(ncoarse_all[i])
    nfine = ncoarse
    para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=itmax, u0=solution(problem, uhat0))
    it, res[:,:,i] = para.run()
    plt.semilogy(range(1,itmax),res[0,:-1,i], label=f"n_coarse = {ncoarse}")

plt.xlabel("$k$")
plt.ylim(top=1e2)
plt.ylim(bottom=1e-10)
plt.yticks([])
# plt.legend(fontsize=fs_legend)
# plt.title("bwe, vary coarse time step")
plt.tight_layout()

plt.savefig("BWEcomplex.pgf")







## Neural network - vary ncoarse
# problem parameters
uadv = 1.0/4    # advection speed
nu = 0.0        # diffusivity parameter
nvar = 1        # number of variables

# Spatial grid
m = 16    # Number of grid points in space
L = 1.0   # Width of spatial domain
xi = np.fft.fftfreq(m)*m*2*np.pi/L

# define spatial operator matrix
D = -1j*xi*uadv - nu*xi**2
D = np.reshape(D, (1,m))
nbneurons = 32

problem = problem_linadv(D, uadv, nu, L, nbneurons)

# initial condition
u0    = problem.get_rand_init(1, L=L)
uhat0 = np.fft.fft(u0, axis=2)
initsol = solution(problem, uhat0)

# temporal specifications
tstart = 0
tend = 16

# parareal parameters
tol = 1e-12              # tolerance
nslices = int(tend)     # time slices
coarse = intNN_linadv         # coarse integrator class
fine = intExact_linadv  # fine integrator class
itmax = nslices         # maximal nb of iterations

dtlist =  np.array((0.0025, 0.010, 0.025, 0.100, 0.2500, 1))
ncoarse_all = (1./dtlist).astype(int)

res = np.zeros((nvar, itmax, len(ncoarse_all)))
plt.figure(figsize=half_figs)
plt.subplot(1,2,1)
for i in range(len(ncoarse_all)):
    ncoarse = int(ncoarse_all[i])
    nfine = ncoarse
    print(ncoarse)
    para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=itmax, u0=solution(problem, uhat0))
    it, res[:,:,i] = para.run()
    plt.semilogy(range(1,itmax),res[0,:-1,i], label=f"$\Delta t = {dtlist[i]}$")

    # print(ncoarse, res[0,:,i])
plt.ylim(top=1e8)
plt.ylim(bottom=1e-10)
plt.legend(fontsize=fs_legend, loc="upper right", ncol=2)
plt.xlabel('$k$')
plt.ylabel('$E^k$')
# plt.title("NN, vary coarse time step")

# problem parameters
uadv = 1.0/4    # advection speed
nu = 0.01        # diffusivity parameter
nvar = 1        # number of variables

# Spatial grid
m = 16    # Number of grid points in space
L = 1.0   # Width of spatial domain
xi = np.fft.fftfreq(m)*m*2*np.pi/L

# define spatial operator matrix
D = -1j*xi*uadv - nu*xi**2
D = np.reshape(D, (1,m))
nbneurons = 32

problem = problem_linadv(D, uadv, nu, L, nbneurons)

# initial condition
u0    = problem.get_rand_init(1, L=L)
uhat0 = np.fft.fft(u0, axis=2)
initsol = solution(problem, uhat0)

# temporal specifications
tstart = 0
tend = 16

# parareal parameters
tol = 1e-12              # tolerance
nslices = int(tend)     # time slices
coarse = intNN_linadv         # coarse integrator class
fine = intExact_linadv  # fine integrator class
itmax = nslices         # maximal nb of iterations

dtlist =  np.array((0.0025, 0.010, 0.025, 0.100, 0.2500, 1))
ncoarse_all = (1./dtlist).astype(int)

res = np.zeros((nvar, itmax, len(ncoarse_all)))
plt.subplot(1,2,2)

for i in range(len(ncoarse_all)):
    ncoarse = int(ncoarse_all[i])
    nfine = ncoarse
    print(ncoarse)
    para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=itmax, u0=solution(problem, uhat0))
    it, res[:,:,i] = para.run()
    plt.semilogy(range(1,itmax),res[0,:-1,i], label=f"n_coarse = {ncoarse}")

    # print(ncoarse, res[0,:,i])

plt.ylim(top=1e8)
plt.ylim(bottom=1e-10)
plt.xlabel('$k$')
plt.yticks([])
plt.tight_layout()

plt.savefig("NNcompl.pgf")

# plt.title("NN, vary coarse time step")

## Neural network - vary architecture

# # parareal parameters
# tol = 1e-12              # tolerance
# nslices = int(tend)     # time slices
# coarse = intNN_linadv   # coarse integrator class
# ncoarse = 2
# nfine = ncoarse
# fine = intExact_linadv  # fine integrator class
# itmax = nslices         # maximal nb of iterations

# nbneurons_all = [4,8,16,32,64,128]
# res = np.zeros((nvar, itmax, len(nbneurons_all)))
# plt.figure()

# for i in range(len(nbneurons_all)):
#     nbneurons = nbneurons_all[i]
#     problem = problem_linadv(D, uadv, nu, L, nbneurons)
#     para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=itmax, u0=solution(problem, uhat0))
#     it, res[:,:,i] = para.run()
#     plt.semilogy(range(1,itmax+1),res[0,:,i], label=f"nb_neurons = {nbneurons}")

# plt.ylim(bottom=1e-10)
# plt.legend()
# plt.title("NN, vary nb neurons")

plt.show()