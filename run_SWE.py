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
from scipy import sparse

np.random.seed(125298133)

half_figs = (5.6,2.2)
fullfull_figs = (7, 8)
ms = 4
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

problem = problem_SWE(D, nlin, h_avg, h_ampl, L, nbneurons = 64)

# temporal specifications
tstart = 0
nslices = 16
dtslice = 10
tend = nslices*dtslice
# nfine = 100
dtfine = 0.1
nfine = round((tend-tstart)/nslices/dtfine)
ncoarse = 1
# dtcoarse = dtslice
# ncoarse = round((tend-tstart)/nslices/dtcoarse)

assert np.sqrt(max(xi**2)*g*h_avg+f**2)*dtfine < 2.8, "Did not start execution because dtfine is too big for stability"

# # Initial data
x  = np.linspace(0, L, m, endpoint=False)
# h0 = np.exp(-(x-0.5*L)**2/(0.2*L)**2)[np.newaxis,np.newaxis,:]*h_ampl
# u0 = np.zeros((1,1,m))
# v0 = np.zeros((1,1,m))
# U0 = np.concatenate((u0, v0, h0), axis=1)
U0 = problem.get_rand_init(1)
uhat0 = np.fft.fft(U0, axis=2)

# plt.figure()
# plt.plot(U0[0,0,:])
# plt.plot(U0[0,1,:])
# plt.plot(U0[0,2,:])
# plt.show()

tol = 1e-6
itmax = nslices
Kiter = [5,10,15]
init_sol = solution(problem, uhat0)

fine=intRK4
coarse=intNN_SWE
intname = "NN"

def run_parareal(uhat, k):
    para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=k, u0=uhat)
    para.run()
    print(f"final max residual is {para.get_final_res()}")
    return para.get_last_end_value().y

yend = np.zeros((3,m), dtype='complex')
for k in range(3):
    yend[k,:] = run_parareal(solution(problem, uhat0), Kiter[k])[0,2,:]

intgcoarse = coarse(tstart, tend, nslices*ncoarse, problem)
intgfine = fine(tstart, tend, nslices*nfine, problem)
ycoarse = intgcoarse.run_last(solution(problem, uhat0))[0,2,:]
yfine = intgfine.run_last(solution(problem, uhat0))[0,2,:]

fig1 = plt.figure(figsize=half_figs)
plt.subplot(1,2,1)
plt.plot(x, np.fft.ifft(ycoarse).real, '-o', color='b', linewidth=1.5, label='Coarse', markersize=ms)
plt.plot(x, np.fft.ifft(yend[0,:]).real,  '-', color='darkred', linewidth=1.5, label='$k='+str(Kiter[0])+'$')
plt.plot(x, np.fft.ifft(yend[1,:]).real,  '-', color='orangered', linewidth=1.5, label='$k='+str(Kiter[1])+'$')
plt.plot(x, np.fft.ifft(yend[2,:]).real,  '-', color='orange', linewidth=1.5, label='$k='+str(Kiter[2])+'$')
plt.plot(x, np.fft.ifft(yfine).real, '--', color='k', linewidth=1.5, label='Fine')
plt.xlim([x[0], x[-1]])
plt.ylim([-50,200])
plt.xlabel('$x$ [m]')
plt.ylabel("$h'$ [m]")
# plt.title(f't=16, ncoarse={ncoarse}, {intname}')
# fig1.gca().tick_params(axis='both')
# plt.legend(loc='upper left', fontsize=fs_legend)
# plt.tight_layout()
# plt.show()
filename = 'swe-para.pgf'
# plt.gcf().savefig(filename, bbox_inches='tight')
# call(["pdfcrop", filename, filename])

xi = np.fft.fftshift(xi)
xi = xi[int(m/2):m]/m
uhat0     = np.fft.fftshift(uhat0)
yend[0,:] = np.around(np.fft.fftshift(yend[0,:]), 10)
yend[1,:] = np.around(np.fft.fftshift(yend[1,:]), 10)
yend[2,:] = np.around(np.fft.fftshift(yend[2,:]), 10)
ycoarse = np.around(np.fft.fftshift(ycoarse), 10)
yfine = np.around(np.fft.fftshift(yfine), 10)

# fig2 = plt.figure(figsize=half_figs)
plt.subplot(1,2,2)
plt.semilogy(xi, np.absolute(ycoarse[int(m/2):m])/m, '-o', color='b', linewidth=1.5, label='Coarse')
plt.semilogy(xi, np.absolute(yfine[int(m/2):m])/m, '--', color='k', linewidth=1.5, label='Exact')
plt.semilogy(xi, np.absolute(yend[0,int(m/2):m])/m, '-', color='darkred', linewidth=1.5, label='$k=$ '+str(Kiter[0]))
plt.semilogy(xi, np.absolute(yend[1,int(m/2):m])/m, '-', color='orangered', linewidth=1.5, label='$k=$ '+str(Kiter[1]))
plt.semilogy(xi, np.absolute(yend[2,int(m/2):m])/m, '-', color='orange', linewidth=1.5, label='$k=$ '+str(Kiter[2]))
plt.xlabel('wave number')
plt.ylabel(r'$|\hat{u}|$')
plt.xticks([0, 1e-4])
# plt.xlim([0.0, 0.3])
plt.ylim(bottom=1e-5)
plt.legend(loc='lower right', fontsize=fs_legend, ncol=2)
plt.tight_layout()
plt.savefig("SWEpara.pgf")
plt.show()

# =============================
# animatie van parareal
# =============================
# para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=itmax-2, u0=solution(problem, uhat0))
# it, res = para.run(plot_all=True)
# print(f"final max residual is {para.get_final_res()} ({it} iterations needed)")
# y = np.fft.ifft(para.get_all_end_values(),axis=2).real

# intgfine = intRK4(tstart, tend, nslices*nfine, problem)
# y_finehat = intgfine.run_all(init_sol)
# y_fine = np.fft.ifft(y_finehat, axis=2).real
# y_finehat = np.fft.fftshift(y_finehat[0,:,:,:], axes=1)
# intgcoarse = intNN_SWE(tstart, tend, nslices*ncoarse, problem)
# y_coarse = np.fft.ifft(intgcoarse.run_all(init_sol), axis=2).real

# resid = np.linalg.norm(y[0,:,:,-1] - y_fine[0,:,:,-1], 2, axis=1)
# print("2-norm error = ", resid)

# uhat0 = np.fft.fftshift(uhat0[0,:,:], axes=1)

# tfine = np.linspace(tstart, tend, nslices*nfine, endpoint=False)
# tcoarse = np.linspace(tstart, tend, nslices*ncoarse, endpoint=False)
# fc_ratio = nfine/ncoarse

# xi = np.fft.fftshift(xi)
# xi = xi[int(m/2):m]/m
# coeff = h_ampl/np.power(2, np.arange(m//2))

# fig1 = plt.figure()
# plt.title('difference parareal - fine')
# plt.semilogy(res[0,:], label="u")
# plt.semilogy(res[1,:], '--', label="v")
# plt.semilogy(res[2,:], ':', label="h")
# plt.legend()

# idx=1
# var=0
# plt.figure()
# plt.subplot(2,1,1)
# plt.title(f"SWE, t={tcoarse[idx]}")
# plt.plot(x, y_coarse[0,var,:,idx], label="coarse")
# plt.plot(x, y_fine[0,var,:,int(idx*nfine/ncoarse)], label="fine")
# plt.legend()
# plt.subplot(2,1,2)
# plt.semilogy(x, np.abs(y_coarse[0,var,:,idx]-y_fine[0,var,:,int(idx*nfine/ncoarse)]))
# plt.ylabel("error coarse - fine")
# plt.xlabel("x")

# var=1
# plt.figure()
# plt.subplot(2,1,1)
# plt.title(f"SWE, t={tcoarse[idx]}")
# plt.plot(x, y_coarse[0,var,:,idx], label="coarse")
# plt.plot(x, y_fine[0,var,:,int(idx*nfine/ncoarse)], label="fine")
# plt.legend()
# plt.subplot(2,1,2)
# plt.semilogy(x, np.abs(y_coarse[0,var,:,idx]-y_fine[0,var,:,int(idx*nfine/ncoarse)]))
# plt.ylabel("error coarse - fine")
# plt.xlabel("x")

# var=2
# plt.figure()
# plt.subplot(2,1,1)
# plt.title(f"SWE, t={tcoarse[idx]}")
# plt.plot(x, y_coarse[0,var,:,idx], label="coarse")
# plt.plot(x, y_fine[0,var,:,int(idx*nfine/ncoarse)], label="fine")
# plt.legend()
# plt.subplot(2,1,2)
# plt.semilogy(x, np.abs(y_coarse[0,var,:,idx]-y_fine[0,var,:,int(idx*nfine/ncoarse)]))
# plt.ylabel("error coarse - fine")
# plt.xlabel("x")

# # plt.figure()
# # idx=1600
# # plt.plot(x, y_fine[0,0,:,idx], label='u')
# # plt.plot(x, y_fine[0,1,:,idx], label='v')
# # plt.plot(x, y_fine[0,2,:,idx], label='h')
# # plt.plot(x, y_coarse[0,1,:,int(idx/fc_ratio)], '--', label='v')
# # plt.plot(x, y_coarse[0,2,:,int(idx/fc_ratio)], '--', label='h')
# # plt.plot(x, y_coarse[0,0,:,int(idx/fc_ratio)], '--', label='u')
# # plt.title(f't={idx*dtfine}')
# # plt.legend()
# # plt.show()

# fs=10

# fig,(ax1,ax2,ax3) = plt.subplots(3, 1)
# fig.set_figheight(8)
# dtfine = tfine[1]-tfine[0]
# # dtcoarse = tcoarse[1]-tcoarse[0]
# intv = int(fc_ratio)
# limmax0 = max(min(np.max(y_fine[0,0,:,:]),10),-10)
# limmin0 = max(min(np.min(y_fine[0,0,:,:]),10),-10)
# limmax1 = max(min(np.max(y_coarse[0,1,:,:]),10),-10)
# limmin1 = max(min(np.min(y_coarse[0,1,:,:]),10),-10)
# limmax2 = max(min(np.max(y_fine[0,2,:,:]),1000),-1000)
# limmin2 = max(min(np.min(y_fine[0,2,:,:]),1000),-1000)
# # limmax=10
# # limmin=-10

# def animate(i):
#     t = i * dtfine * intv
#     ax1.clear()
#     ax1.set_title('t={:.2f}, {}, ncoarse={}'.format(t, coarsename, ncoarse))
#     # place a text box in upper left in axes coords
#     ax1.set_ylabel('u [m/s]', fontsize=fs)
#     ax2.clear()
#     ax2.set_ylabel('v [m/s]', fontsize=fs)
#     ax3.clear()
#     ax3.set_ylabel('h [m]', fontsize=fs)
#     ax3.set_xlabel('x [m]') 
#     ax1.plot(x, y_fine[0, 0, :, i * intv],'--', color='k', linewidth=1.5, label='Fine')
#     ax1.plot(x, y_coarse[0, 0, :,  int(i * intv / fc_ratio)],'-o', color='b', linewidth=1.5, label='Coarse', markevery=(3,7), markersize=fs/2)    
#     ax1.plot(x, y[0, 0, :, i * intv],'-s', color='r', linewidth=1.5, label='Parareal k='+str(it), markevery=(1,3), mew=1.0, markersize=fs/2)
#     ax2.plot(x, y_fine[0, 1, :, i * intv],'--', color='k', linewidth=1.5, label='Fine')
#     ax2.plot(x, y_coarse[0, 1, :,  int(i * intv / fc_ratio)],'-o', color='b', linewidth=1.5, label='Coarse', markevery=(3,7), markersize=fs/2)    
#     ax2.plot(x, y[0, 1, :, i * intv],'-s', color='r', linewidth=1.5, label='Parareal k='+str(it), markevery=(1,3), mew=1.0, markersize=fs/2)
#     ax3.plot(x, y_fine[0, 2, :, i * intv],'--', color='k', linewidth=1.5, label='Fine')
#     ax3.plot(x, y_coarse[0, 2, :,  int(i * intv / fc_ratio)],'-o', color='b', linewidth=1.5, label='Coarse', markevery=(3,7), markersize=fs/2)    
#     ax3.plot(x, y[0, 2, :, i * intv],'-s', color='r', linewidth=1.5, label='Parareal k='+str(it), markevery=(1,3), mew=1.0, markersize=fs/2)
#     ax1.set_ylim([limmin0*1.2, limmax0*1.2])
#     ax2.set_ylim([limmin1*1.2, limmax1*1.2])
#     ax3.set_ylim([limmin2*1.2, limmax2*1.2])
#     ax1.legend()

# anim = animation.FuncAnimation(fig, animate, frames = int(np.shape(y_fine)[3]/intv),
#      interval = 300, repeat = False)

# # flnm = os.path.join(os.path.dirname(__file__), "../figs_linadv/SWElin-NN-k23.gif")
# # anim.save(flnm, dpi=100, writer=animation.PillowWriter(fps=10))

# plt.show()