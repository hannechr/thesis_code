import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))
import numpy as np
from problem_linadv_m import problem_linadv
from intExact_linadv_m import intExact_linadv
from intNN_linadv_m import intNN_linadv
from solution_m import solution
from intRK4_m import intRK4
import matplotlib
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

half_figs = (5.6,4.5)
fullfull_figs = (7, 8)

fs_legend = 7

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# PARAMETERS
dtlist =  np.array((0.0025, 0.010, 0.025, 0.100, 0.2500, 1, 2.5))
t_end = 15
ntests = 256
dtref = 0.0025
nsteps_ref = int(t_end/dtref)

uadv = 1/4
nu = 0.0

# Spatial grid
m = 16    # Number of grid points in space
L = 1.0  # Width of spatial domain

# Wavenumber "grid"
xi = np.fft.fftfreq(m)*m*2*np.pi/L

# define spatial operator matrix
D = -1j*xi*uadv - nu*xi**2
D = np.reshape(D, (1,m))

problem = problem_linadv(D, uadv, nu, L)

u0    = problem.get_rand_init(ntests, L=L) # ntests*nvar*nspat
u0hat = np.fft.fft(u0, axis=2)

init = solution(problem, u0hat)
intref = intExact_linadv(0, t_end, nsteps_ref, problem)
refhat = intref.run_all(init)

error = []
simhat_list = []
Em_list = []
Ep_list = []
intNN_list = []

for i in range(len(dtlist)):
    dt = dtlist[i]
    print(f"working on dt = {dt}")
    nsteps = int(t_end/dt)
    intNN = intNN_linadv(0, t_end, nsteps, problem)
    intNN_list.append(intNN)
    simhat = intNN.run_all(init)
    simhat_list.append(simhat)
    step_gap_ref = int(dt/dtref)
    refhat_ts = refhat[:,:,:,np.arange(0,nsteps_ref,step_gap_ref)]
    ref_ts = np.fft.ifft(refhat_ts, axis=2)
    sim = np.fft.ifft(simhat, axis=2)
    Epm = np.max(abs(ref_ts - sim), axis=(1,2))
    Em = np.max(Epm, axis=1)
    Em_list.append(Em)
    # print(Em)
    Ep = np.mean(Epm, axis=0)
    Ep_list.append(Ep)
    # print(Ep)
    E = np.mean(Em)
    error.append(E)

colors = list(mcolors.TABLEAU_COLORS.keys())

# plot Ep for different time steps
fig = plt.figure(figsize=half_figs)

gs = fig.add_gridspec(2, hspace=0.1)
axs = gs.subplots(sharex=True)
for i in [0,1,2,3,4,5,6]:
    col = colors[i]
    dt = dtlist[i]
    t = np.arange(0, t_end, dt)
    axs[0].loglog(t[1:], Ep_list[i][1:], color=col, label=f"$\Delta t={dtlist[i]}$")

axs[0].loglog([2e-3, 1e1],[2e-4, 1e0], 'k--', label="$\mathcal{O}(N)$")
# axs[0].legend(loc="upper left", fontsize=fs_legend, ncol=2)
axs[0].set_ylabel("$\\nu=0.0$")
# plt.tight_layout
uadv = 1/4
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

u0    = problem.get_rand_init(ntests, L=L) # ntests*nvar*nspat
u0hat = np.fft.fft(u0, axis=2)

init = solution(problem, u0hat)
intref = intExact_linadv(0, t_end, nsteps_ref, problem)
refhat = intref.run_all(init)

error = []
simhat_list = []
Em_list = []
Ep_list = []
intNN_list = []

for i in range(len(dtlist)):
    dt = dtlist[i]
    print(f"working on dt = {dt}")
    nsteps = int(t_end/dt)
    intNN = intNN_linadv(0, t_end, nsteps, problem)
    intNN_list.append(intNN)
    simhat = intNN.run_all(init)
    simhat_list.append(simhat)
    step_gap_ref = int(dt/dtref)
    refhat_ts = refhat[:,:,:,np.arange(0,nsteps_ref,step_gap_ref)]
    ref_ts = np.fft.ifft(refhat_ts, axis=2)
    sim = np.fft.ifft(simhat, axis=2)
    Epm = np.max(abs(ref_ts - sim), axis=(1,2))
    Em = np.max(Epm, axis=1)
    Em_list.append(Em)
    # print(Em)
    Ep = np.mean(Epm, axis=0)
    Ep_list.append(Ep)
    # print(Ep)
    E = np.mean(Em)
    error.append(E)

for i in [0,1,2,3,4,5,6]:
    col = colors[i]
    dt = dtlist[i]
    t = np.arange(0, t_end, dt)
    axs[1].loglog(t[1:], Ep_list[i][1:], color=col, label=f"$\Delta t={dtlist[i]}$")

axs[1].loglog([2e-3, 1e1],[2e-4, 1e0], 'k--', label="$\mathcal{O}(N)$")
axs[1].set_ylabel("$\\nu=0.01$")
axs[1].set_xlabel("$t$[s]")
axs[1].legend(loc="upper left", fontsize=fs_legend, ncol=2)

plt.subplots_adjust(bottom=0.11)
plt.tight_layout()
plt.savefig("linadverrorNN.pgf")
plt.show()


# dtshow = 0.1
# nshow = int(t_end/dtshow)
# dtfine = dtref
# intv = 1
# fcratio = int(dtshow/dtref)
# x = np.linspace(0,L, m, endpoint=False)

# initan = u0hat[[0],:,:]
# initann = solution(problem, initan)
# intNN = intNN_linadv(0, t_end, nshow, problem)
# sol = intNN.run_all(initann)

# y_fine = np.fft.ifft(refhat, axis=2)[0,0,:,:].real
# y_coarse = np.fft.ifft(sol, axis=2)[0,0,:,:].real

# fig, (ax1, ax2) = plt.subplots(2, 1)

# def animate(i):
#     ax1.clear()
#     t = i * dtfine * intv * fcratio
#     ax1.plot(x, y_fine[:, i * intv * fcratio],'k--', linewidth=1.5, label='fine')
#     ax1.plot(x, y_coarse[:, i * intv],'b', linewidth=1.5, label='coarse')

#     # ax1.set_ylim([-1.5,1.5])
#     ax1.set_xlabel('x [m]')
#     ax1.set_ylabel('')
#     ax1.set_title('t={:.2f}'.format(t))
#     ax1.legend()
#     ax2.clear()
#     ax2.semilogy(x, abs(y_coarse[:, i * intv ]-y_fine[:, i * intv * fcratio])+1e-15, label='error coarse')
#     ax2.set_ylim([1e-6,1e1])
#     ax2.set_xlabel('x[m]')
#     ax2.set_ylabel('error coarse')

# anim = animation.FuncAnimation(fig, animate, frames = int(np.shape(y_coarse)[1]/intv),
#      interval = 100, repeat = False)


plt.show()
