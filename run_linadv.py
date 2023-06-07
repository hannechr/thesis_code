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
from intFWE_m import intFWE
from intExact_linadv_m import intExact_linadv
from intNN_linadv_m import intNN_linadv
# from intNN_linadv_m import intNN_linadv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

np.random.seed(125293)

half_figs = (5.6,2.1)
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


# advection speed
uadv = 1.0/4

# diffusivity parameter
nu = 0.01

# Spatial grid
m = 16    # Number of grid points in space
L = 1.0  # Width of spatial domain

# Wavenumber "grid"
xi = np.fft.fftfreq(m)*m*2*np.pi/L

# define spatial operator matrix
D = -1j*xi*uadv - nu*xi**2
D = np.reshape(D, (1,m))

problem = problem_linadv(D, uadv, nu, L)

# temporal specifications
tstart = 0
tend = 16
nslices = int(tend)
ncoarse = 2
nfine = ncoarse

# # Initial data
x     = np.linspace(0, L, m, endpoint=False)
# u0    = np.exp(-(x-0.5*L)**2/(L/4)**2)[np.newaxis, np.newaxis, :]
u0    = problem.get_rand_init(1, L=L)
uhat0 = np.fft.fft(u0, axis=2)

tol = 1e-16
itmax = nslices
Kiter = [5,10,15]
init_sol = solution(problem, uhat0)

fine=intExact_linadv
coarse=intBWE
intname = "BWE"
# coarse = intNN_linadv
# intname = "NN"

def run_parareal(uhat, k):
    para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=k, u0=uhat)
    para.run()
    print(f"final max residual is {para.get_final_res()}")
    return para.get_last_end_value().y

yend = np.zeros((3,m), dtype='complex')
for k in range(3):
    yend[k,:] = run_parareal(init_sol, Kiter[k])[0,0,:]

intgcoarse = coarse(tstart, tend, nslices*ncoarse, problem)
intgfine = fine(tstart, tend, nslices*ncoarse, problem)
ycoarse = intgcoarse.run_last(solution(problem, uhat0))[0,0,:]
yfine = intgfine.run_last(solution(problem, uhat0))[0,0,:]

fig1 = plt.figure(figsize=half_figs)
plt.subplot(1,2,1)
plt.plot(x, np.fft.ifft(ycoarse).real, '-o', color='b', linewidth=1.5, label='Coarse', markersize=ms)
plt.plot(x, np.fft.ifft(yend[0,:]).real,   color='darkred', linewidth=1.5, label='$k='+str(Kiter[0])+'$')
plt.plot(x, np.fft.ifft(yend[1,:]).real,   color='orangered', linewidth=1.5, label='$k='+str(Kiter[1])+'$')
plt.plot(x, np.fft.ifft(yend[2,:]).real,   color='orange', linewidth=1.5, label='$k='+str(Kiter[2])+'$')
plt.plot(x, np.fft.ifft(yfine).real, '--', color='k', linewidth=1.5, label='Fine')
plt.xlim([x[0], x[-1]])
# plt.ylim([-1.1, 1.4])
plt.xlabel('$x$ [m]')
plt.ylabel('$u$ [m]')
# plt.title(f't=16, ncoarse={ncoarse}, {intname}')
fig1.gca().tick_params(axis='both')
# plt.legend(loc='upper left', fontsize=fs_legend)
# plt.tight_layout()
# plt.show()
filename = 'linadv-para.pgf'
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
plt.semilogy(xi, np.absolute(ycoarse[int(m/2):m])/m, '-o', color='b', linewidth=1.5, label='Coarse', markersize=ms)
plt.semilogy(xi, np.absolute(yfine[int(m/2):m])/m, '--', color='k', linewidth=1.5, label='Exact')
plt.semilogy(xi, np.absolute(yend[0,int(m/2):m])/m, '-', color='darkred', linewidth=1.5, label='$k=$ '+str(Kiter[0]))
plt.semilogy(xi, np.absolute(yend[1,int(m/2):m])/m, '-', color='orangered', linewidth=1.5, label='$k=$ '+str(Kiter[1]))
plt.semilogy(xi, np.absolute(yend[2,int(m/2):m])/m, '-', color='orange', linewidth=1.5, label='$k=$ '+str(Kiter[2]))
plt.xlabel('wave number', labelpad=0)
plt.ylabel(r'$|\hat{u}|$', labelpad=0)
# plt.xticks([0.0, 0.1, 0.3], fontsize=fs)
# plt.xlim([0.0, 0.3])
plt.yticks([1e0, 1e-3, 1e-6, 1e-9, 1e-12])
plt.legend(loc='lower right', fontsize=fs_legend)
plt.tight_layout()
plt.show()


plt.savefig("BWEcoarse2diff.pgf")


# para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=15, u0=init_sol)
# it,res = para.run(plot_all=True, xi=xi)
# print(res)

# plt.subplot(2,5,3)
# plt.xlabel("$x$ [m]", fontsize=9, labelpad=0)
# plt.subplot(2,5,8)
# plt.xlabel("wave number [m$^{-1}$]", fontsize=9)
# plt.gcf().subplots_adjust(hspace=0.4, bottom=0.15)
# plt.savefig('badgeneralis.pgf')
# plt.show()

# # =============================
# # hier animatie van parareal
# # =============================

# para = parareal(tstart, tend, nslices, fine, coarse, nfine, ncoarse, tol, iter_max=itmax, u0=init_sol)
# it, res = para.run()
# print(f"final max residual is {para.get_final_res()}")
# y = np.fft.ifft(para.get_all_end_values(),axis=2).real

# intfine = fine(0, tend, nfine*nslices, problem)
# intcoarse = coarse(0, tend, ncoarse*nslices, problem)

# y_fine = np.fft.ifft(intfine.run_all(init_sol), axis=2).real
# y_coarse = np.fft.ifft(intcoarse.run_all(init_sol), axis=2).real

# tfine = np.linspace(tstart, tend, nslices*nfine, endpoint=False)
# tcoarse = np.linspace(tstart, tend, nslices*ncoarse, endpoint=False)



# print(res)
# fig1 = plt.figure()
# plt.title('parareal residual')
# plt.semilogy(range(1,itmax+1),res[0,:], label="u")
# plt.ylim(bottom=1e-10)
# plt.legend()

# fig, (ax1, ax2) = plt.subplots(2, 1)
# dtfine = tfine[1]-tfine[0]
# fc_ratio = nfine/ncoarse
# intv = int(fc_ratio)
# limmax = max(min(np.max(y_fine[0,:,:,:]),10),-10)
# limmin = max(min(np.min(y_fine[0,:,:,:]),10),-10)
# # limmax=10
# # limmin=-10

# def animate(i):
#     ax1.clear()
#     t = i * dtfine * intv
#     ax1.plot(x, y_fine[0, 0, :, i * intv],'k--', linewidth=1.5, label='fine')
#     ax1.plot(x, y[0, 0, :, i * intv ],'r', linewidth=1.5, label='parareal')
#     ax1.plot(x, y_coarse[0, 0, :, int(i * intv / fc_ratio)],'b', linewidth=1.5, label='coarse')

#     # ax1.set_ylim([limmin*1.1,limmax*1.1])
#     ax1.set_xlabel('x [m]')
#     ax1.set_ylabel('')
#     ax1.set_title('t={:.2f}, ncoarse={}'.format(t, ncoarse))
#     ax1.legend()
#     ax2.clear()
#     ax2.semilogy(x, abs(y_coarse[0, 0, :, int(i * intv / fc_ratio)]-y_fine[0, 0, :, i * intv])+1e-15, label='error coarse')
#     ax2.set_ylim([1e-6,1e1])
#     ax2.set_xlabel('x[m]')
#     ax2.set_ylabel('error coarse')

# anim = animation.FuncAnimation(fig, animate, frames = int(np.shape(y)[3]/intv),
#      interval = 100, repeat = False)

# # flnm = sys.path.append(os.path.join(os.path.dirname(__file__), "../figs_linadv/BWE.gif"))
# # anim.save(flnm, dpi=100, writer=animation.PillowWriter(fps=15))

# plt.show()