import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))
import numpy as np
from problem_oscillator_m import problem_oscillator
from problem_vdp_m import problem_vdp
from intExact_oscillator_m import intExact_oscillator
from intNN_oscillator_m import intNN_oscillator
from intNN_vdp_m import intNN_vdp
from solution_m import solution
from intRK4_m import intRK4
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

np.random.seed(124896)

half_figs = (5.1,2.3)
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
ntests = 1024
dtref = 0.0025
nsteps_ref = int(t_end/dtref)
a = 1

r0 = np.random.uniform(1.05,2.2, (ntests, 1, 1))
phi0 = np.random.uniform(-np.pi, np.pi, (ntests, 1, 1))
init = np.concatenate((r0*np.sin(phi0),r0*np.cos(phi0)), axis=1)

osc = problem_oscillator(a)
osc_init = solution(osc, init.astype(np.double))
osc_intref = intExact_oscillator(0, t_end, nsteps_ref, osc)
osc_ref = osc_intref.run_all(osc_init).real

osc_error = []
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
    osc_error.append(E)

vdp_ref_list = []
vdp_error = []
vdp_sim_list = []
vdp_Em_list = []
vdp_Ep_list = []
vdp_intNN_list = []

mulist = [0.5, 1, 2, 4]
for k in range(len(mulist)):
    mu = mulist[k]
    prob = problem_vdp(mu)
    vdp_init = solution(prob, init)

    print(f"generating reference solution, mu= {mu}")
    vdp_intref = intRK4(0, t_end, nsteps_ref, prob)
    vdp_ref = vdp_intref.run_all(vdp_init).real
    vdp_ref_list.append(vdp_ref)
    vdp_error.append([])
    vdp_sim_list.append([])
    vdp_Em_list.append([])
    vdp_Ep_list.append([])
    vdp_intNN_list.append([])
    for i in range(len(dtlist)):
        dt = dtlist[i]


        print(f"working on dt = {dt}")
        nsteps = int(t_end/dt)
        vdp_intNN = intNN_vdp(0, t_end, nsteps, prob)
        vdp_intNN_list[k].append(vdp_intNN)
        vdp_sim = vdp_intNN.run_all(vdp_init)
        vdp_sim_list[k].append(vdp_sim)
        step_gap_ref = int(dt/dtref)
        vdp_ref_ts = vdp_ref[:,:,:,np.arange(0,nsteps_ref,step_gap_ref)]
        vdp_Epm = np.max(abs(vdp_ref_ts - vdp_sim), axis=(1,2))
        vdp_Em = np.max(vdp_Epm, axis=1)
        vdp_Em_list[k].append(vdp_Em)
        vdp_Ep = np.mean(vdp_Epm, axis=0)
        vdp_Ep_list[k].append(vdp_Ep)
        vdp_E = np.mean(vdp_Em)
        vdp_error[k].append(vdp_E)




colors = list(mcolors.TABLEAU_COLORS.keys())
k=1
# i=6
# plt.figure()
# plt.title(f"dt={dtlist[i]}")
# plt.scatter(init[:,0,0], init[:,1,0], c=vdp_Em_list[k][i])
# plt.plot(vdp_ref_list[k][0,0,0,1000:], vdp_ref_list[k][0,1,0,1000:],'--k')
# plt.colorbar()

i=3
plt.figure()
plt.title(f"dt={dtlist[i]}")
plt.scatter(init[:,0,0], init[:,1,0], c=np.log(vdp_Em_list[k][i]))
plt.plot(vdp_ref_list[k][0,0,0,1000:], vdp_ref_list[k][0,1,0,1000:],'--k')
plt.colorbar()
plt.tight_layout()
# plt.show()

# i=0
# plt.figure()
# plt.title(f"dt={dtlist[i]}")
# plt.scatter(init[:,0,0], init[:,1,0], c=vdp_Em_list[k][i])
# plt.plot(vdp_ref_list[k][0,0,0,1000:], vdp_ref_list[k][0,1,0,1000:],'--k')
# plt.colorbar()

# plt.show()


# plot simulation of run m for different time steps, for mu=2 (k=2).
m=0
k=2
fig = plt.figure(figsize=(5.6, 3.5))
gs = fig.add_gridspec(2, hspace=0.15)
axs = gs.subplots(sharex=True)
for i in [0,2,4,6]:
    col = colors[i]
    dt = dtlist[i]
    t = np.arange(0, t_end, dt)
    plt.subplot(2,1,1)
    plt.plot(t, (vdp_sim_list[k][i])[m,0,0,:], color=col, alpha=1, label=f"$\Delta t={dtlist[i]}$")
    plt.subplot(2,1,2)
    plt.plot(t, (vdp_sim_list[k][i])[m,1,0,:], color=col, alpha=1, label=f"$\Delta t={dtlist[i]}$")

tref = np.arange(0, t_end, dtref)
plt.subplot(2,1,1)
plt.plot(tref, vdp_ref_list[k][m,0,0,:], '--', color='black', label="exact")
plt.ylabel("$u$")
plt.subplot(2,1,2)
plt.plot(tref, vdp_ref_list[k][m,1,0,:], '--', color='black', label="exact")
plt.legend(loc="upper left", fontsize=fs_legend, ncol=2)
plt.ylabel("$v$")
plt.xlabel("$t$ [s]")
plt.subplots_adjust(bottom=0.13)
plt.savefig("vdp-sim.pgf")


# plot training for different networks
plt.figure(figsize=half_figs)
for i in [0,2,4,6]:
    col = colors[i]
    plt.loglog(vdp_intNN_list[k][i].net.loss, color=col, alpha=1, label=f"$\Delta t={dtlist[i]}$")

plt.xlabel("iteration")
plt.ylabel("loss")
plt.subplots_adjust(bottom=0.2)
plt.legend(loc="upper right", fontsize=fs_legend)



# plot Ep for different time steps
# print(vdp_Ep_list[k][6])
plt.figure(figsize=half_figs)
for i in [0,2,4,6]:
    col = colors[i]
    dt = dtlist[i]
    t = np.arange(0, t_end, dt)
    plt.loglog(t[1:], vdp_Ep_list[k][i][1:], color=col, alpha=1, label=f"$\Delta t={dtlist[i]}$")

plt.loglog([5e-4, 1e1],[5e-5, 1e0], 'k--', label="$O(N)$")
plt.legend(loc="upper left", fontsize=fs_legend)
plt.xlabel("$t$ [s]")
plt.ylabel("$E_p$")
plt.subplots_adjust(bottom=0.2)
plt.savefig("vdp-err.pgf")


# plot E for different mu, for different time steps
plt.figure(figsize=half_figs)
mulst = mulist.copy()
mulst.insert(0,0)
for i in [0, 2, 4, 6]:
    col = colors[i]
    dt = dtlist[i]
    vdp_errork = [vdp_error[j][i] for j in range(len(mulist))]
    vdp_errork.insert(0,osc_error[i])
    plt.semilogy(mulst, vdp_errork, color=col, alpha=1, label=f"$\Delta t={dtlist[i]}$")


plt.legend(fontsize=fs_legend)
plt.xlabel("$\mu$")
plt.ylabel("E")
plt.subplots_adjust(bottom=0.2)
plt.savefig("mu-diff.pgf")


plt.show()
