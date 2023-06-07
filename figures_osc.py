import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))
import numpy as np
from problem_oscillator_m import problem_oscillator
from intExact_oscillator_m import intExact_oscillator
from intNN_oscillator_m import intNN_oscillator
from solution_m import solution
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

np.random.seed(129266)

half_figs = (5.1,2.3)
legend_fs = 7

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
ntests = 128
dtref = 0.0025
nsteps_ref = int(t_end/dtref)
a = 1

r0 = np.random.uniform(1.25, 2.250, (ntests, 1, 1))
phi0 = np.random.uniform(-np.pi, np.pi, (ntests, 1, 1))
init = np.concatenate((r0*np.sin(phi0),r0*np.cos(phi0)), axis=1)
osc = problem_oscillator(a)
osc_init = solution(osc, init.astype(np.double))
osc_intref = intExact_oscillator(0, t_end, nsteps_ref, osc)
osc_ref = osc_intref.run_all(osc_init).real

error = []
osc_sim_list = []
Em_list = []
Ep_list = []
osc_intNN_list = []

for i in range(len(dtlist)):
    dt = dtlist[i]
    print(f"working on dt = {dt}")
    nsteps = int(t_end/dt)
    osc_intNN = intNN_oscillator(0, t_end, nsteps, osc)
    osc_intNN_list.append(osc_intNN)
    osc_sim = osc_intNN.run_all(osc_init)
    osc_sim_list.append(osc_sim)
    step_gap_ref = int(dt/dtref)
    osc_ref_ts = osc_ref[:,:,:,np.arange(0,nsteps_ref,step_gap_ref)]
    Epm = np.max(abs(osc_ref_ts - osc_sim), axis=(1,2))
    Em = np.max(Epm, axis=1)
    Ep = np.mean(Epm, axis=0)
    Ep_list.append(Ep)
    E = np.mean(Em)
    error.append(E)
    


# plot simulation of run m for different time steps.
colors = list(mcolors.TABLEAU_COLORS.keys())
m=0
fig = plt.figure(figsize=(5.6, 3.5))
gs = fig.add_gridspec(2, hspace=0.15)
axs = gs.subplots(sharex=True)
for i in [0,2,4,6]:
    col = colors[i]
    dt = dtlist[i]
    t = np.arange(0, t_end, dt)
    axs[0].plot(t, (osc_sim_list[i])[m,0,0,:], color=col, label=f"$\Delta t={dtlist[i]}$")
    axs[1].plot(t, (osc_sim_list[i])[m,1,0,:], color=col, label=f"$\Delta t={dtlist[i]}$")

tref = np.arange(0, t_end, dtref)
plt.subplot(2,1,1)
plt.plot(tref, osc_ref[m,0,0,:], '--', color='black', label="exact")
plt.ylabel("$u$")
plt.subplot(2,1,2)
plt.plot(tref, osc_ref[m,1,0,:], '--', color='black', label="exact")
plt.legend(loc="lower left", fontsize=legend_fs, ncol=2)
plt.ylabel("$v$")
plt.xlabel("$t$ [s]")
# plt.tight_layout()
plt.subplots_adjust(bottom=0.13)
plt.savefig('osc-sim.pgf')


# plot training for different networks
plt.figure(figsize=half_figs)
for i in [0,2,4,6]:
    col = colors[i]
    plt.loglog(osc_intNN_list[i].net.loss, color=col, alpha=0.7, label=f"$\Delta t={dtlist[i]}$")

plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(loc="upper right", fontsize=legend_fs)
plt.subplots_adjust(bottom=0.2)
# plt.tight_layout()
plt.savefig('osc-training.pgf')


# plot Ep for different time steps
plt.figure(figsize=half_figs)
for i in [0,2,4,6]:
    col = colors[i]
    dt = dtlist[i]
    t = np.arange(0, t_end, dt)
    plt.loglog(t[1:], Ep_list[i][1:], color=col, alpha=0.7, label=f"$\Delta t={dtlist[i]}$")

plt.loglog([2e-3, 1e1],[2e-4, 1e0], 'k--', label="$\mathcal{O}(N)$")
plt.legend(loc="upper left", fontsize=legend_fs)
plt.xlabel("$t$ [s]")
plt.ylabel("$E_p$")
plt.subplots_adjust(bottom=0.2)
# plt.tight_layout()
plt.savefig("osc-err.pgf")


plt.show()