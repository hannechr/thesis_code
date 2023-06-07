import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))
import numpy as np
from problem_vdp_m import problem_vdp
from problem_oscillator_m import problem_oscillator
from intExact_oscillator_m import intExact_oscillator
from intExact_linadv_m import intExact_linadv
from intRK4_m import intRK4
from intNN_oscillator_m import intNN_oscillator
from intNN_vdp_m import intNN_vdp
from intNN_linadv_m import intNN_linadv
from solution_m import solution
from problem_linadv_m import problem_linadv
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

# PARAMETERS
dtlist =  np.array((0.0025, 0.010, 0.025, 0.100, 0.2500, 1, 2.5))
mulist = [0.5, 1, 2, 4]
t_end = 16
ntests = 128
dtref = 0.001
nsteps_ref = int(t_end/dtref)
muplot = 2

# compare different timesteps for one run oscillator
fig_osc_sim = plt.figure()
# compare different timesteps for one run van der pol
fig_vdp_sim = plt.figure()
# compare different timesteps mean error oscillator
fig_err_osc = plt.figure()
# compare different timesteps mean error van der pol
fig_err_vdp = plt.figure()
# for different mu, compare avg end error for different dt
fig_nlin_max = plt.figure()
fig_nlin_mse = plt.figure()
# for training loss evolution
fig_loss = plt.figure()
fig_loss_osc = plt.figure()

colors2=iter(['-','--',':','-.'])

for k in range(len(mulist)):
    mu = mulist[k]
    print(f"====== mu: {mu} ======")
    vdp = problem_vdp(mu)

    # ACQUIRE REFERENCE SOLUTION : ntests x nvar x nspat x nsteps
    vdp_ref_file = os.path.join(os.path.dirname(__file__), f"sol-prob/refs/vdp_ref_{dtref}_{t_end}s_{ntests}tests_mu_{mu}.npy")
    try: # if reference solution exists already
        with open(vdp_ref_file, 'rb') as f:
            vdp_ref = np.load(f)
        print("test data loaded")

    except: # generate reference solution
        print("no test data exists yet")
        # create initial values:
        print("generating initial values")
        r0 = np.random.uniform(0.6, 1.4, (ntests, 1, 1))
        phi0 = np.random.uniform(-np.pi, np.pi, (ntests, 1, 1))
        init = np.concatenate((r0*np.sin(phi0),r0*np.cos(phi0)), axis=1).astype(np.double)
        vdp_init = solution(vdp, init)

        # integrate:       # vdp: RK4 dtref
        print("generating reference solution")
        vdp_intref = intRK4(             0, t_end, nsteps_ref, vdp)
        vdp_ref = vdp_intref.run_all(vdp_init).real
        with open(vdp_ref_file, 'wb') as f:
            np.save(f, vdp_ref)
    vdp_init = solution(vdp, vdp_ref[:,:,:,0])

    vdp_max_error = []
    vdp_mse_error = []

    # TRAIN NETWORKS & FORECAST
    print("training networks & perform forecasting")
    tref = np.arange(0,t_end, dtref)
    colors=iter(plt.cm.rainbow(np.linspace(0, 1, len(dtlist))))
    for i in range(len(dtlist)):
        dt = dtlist[i]
        rgb = next(colors)
        print(f"working on dt = {dt}")
        nsteps = int(t_end/dt)
        vdp_sim = np.zeros((ntests, 2, 1, nsteps))
        vdp_NN = intNN_vdp(0, t_end, nsteps, vdp, smart=True)
        vdp_sim[:,:,:,:] = vdp_NN.run_all(vdp_init)
        t = np.arange(0, t_end, dt)

        assert dt/dtref % 1 == 0, "dt should be multiple of reference time step dtref"
        step_gap_ref = int(dt/dtref)
        vdp_ref_ts = vdp_ref[:,:,:,np.arange(0,nsteps_ref,step_gap_ref)]
        assert np.shape(vdp_ref_ts) == np.shape(vdp_sim)
        vdp_diff = abs(vdp_ref_ts - vdp_sim)
        vdp_mean = np.mean(vdp_diff, axis=0)
        vdp_max = np.max(vdp_diff, axis=3)
        vdp_mse_error.append(np.mean(np.mean(np.mean(vdp_diff**2, axis=0), axis=2)))
        vdp_max_error.append(np.mean(vdp_max))

        # PLOT SECTION
        # ntests x nvar x nspat x nsteps

        ex = 0

        if mu == muplot:

            plt.figure(fig_vdp_sim)
            plt.subplot(2,1,1)
            plt.plot(t, vdp_sim[ex,0,0,:], alpha=0.8, color=rgb, label=f'dt={dt}')
            plt.subplot(2,1,2)
            plt.plot(t, vdp_sim[ex,1,0,:], alpha=0.8, color=rgb, label=f'dt={dt}')    

            plt.figure(fig_err_vdp)
            plt.subplot(2,1,1)
            plt.loglog(t[1:], vdp_mean[0,0,1:], alpha=0.8, color=rgb, label=f'dt={dt}')
            plt.subplot(2,1,2)
            plt.loglog(t[1:], vdp_mean[1,0,1:], alpha=0.8, color=rgb, label=f'dt={dt}')

            plt.figure(fig_loss)
            plt.semilogy(vdp_NN.net.loss, color=rgb, alpha=0.5, label=f'dt={dt}')


    if mu == muplot:

        plt.figure(fig_vdp_sim)
        plt.subplot(2,1,1)
        plt.title("Van der Pol oscillator")
        plt.plot(tref, vdp_ref[ex,0,0,:],'k',label='reference')
        plt.ylabel("1st component")
        plt.subplot(2,1,2)
        plt.plot(tref, vdp_ref[ex, 1,0,:],'k',label='reference')
        plt.legend()
        plt.xlabel("time (s)")
        plt.ylabel("2nd component")


    rgb2 = next(colors2)
    plt.figure(fig_nlin_max)
    plt.loglog(dtlist, vdp_max_error, rgb2, alpha=0.8, label=f"mu={mu}")

    plt.figure(fig_nlin_mse)
    plt.loglog(dtlist, vdp_mse_error, rgb2, alpha=0.8, label=f"mu={mu}")

# ================================================================================
r0 = np.random.uniform(0.6, 1.4, (ntests, 1, 1))
phi0 = np.random.uniform(-np.pi, np.pi, (ntests, 1, 1))
init = np.concatenate((r0*np.sin(phi0),r0*np.cos(phi0)), axis=1)
osc = problem_oscillator(1)
osc_init = solution(osc, init.astype(np.double))
osc_intref = intExact_oscillator(0, t_end, nsteps_ref, osc)
osc_ref = osc_intref.run_all(osc_init).real
osc_init = solution(osc, osc_ref[:,:,:,0])
osc_max_error = []
osc_mse_error = []

colors=iter(plt.cm.rainbow(np.linspace(0, 1, len(dtlist))))

for i in range(len(dtlist)):
    dt = dtlist[i]
    rgb = next(colors)
    print(f"working on dt = {dt}")
    nsteps = int(t_end/dt)
    osc_sim = np.zeros((ntests, 2, 1, nsteps))
    osc_NN = intNN_oscillator(0, t_end, nsteps, osc)
    osc_sim[:,:,:,:] = osc_NN.run_all(osc_init)
    t = np.arange(0, t_end, dt)
    step_gap_ref = int(dt/dtref)
    osc_ref_ts = osc_ref[:,:,:,np.arange(0,nsteps_ref,step_gap_ref)]
    osc_diff = abs(osc_ref_ts - osc_sim)
    osc_mean = np.mean(osc_diff, axis=0)
    osc_max = np.max(osc_diff, axis=3)
    osc_mse_error.append(np.mean(np.mean(np.mean(osc_diff**2, axis=0), axis=2)))
    osc_max_error.append(np.mean(osc_max))

    plt.figure(fig_osc_sim)
    plt.subplot(2,1,1)
    plt.plot(t, osc_sim[ex,0,0,:], alpha=0.8, color=rgb, label=f'dt={dt}')
    plt.subplot(2,1,2)
    plt.plot(t, osc_sim[ex,1,0,:], alpha=0.8, color=rgb, label=f'dt={dt}')

    plt.figure(fig_err_osc)
    plt.subplot(2,1,1)
    plt.loglog(t[1:], osc_mean[0,0,1:], alpha=0.8, color=rgb, label=f'dt={dt}')
    plt.subplot(2,1,2)
    plt.loglog(t[1:], osc_mean[1,0,1:], alpha=0.8, color=rgb, label=f'dt={dt}')

    plt.figure(fig_loss_osc)
    plt.semilogy(osc_NN.net.loss, color=rgb, alpha=0.5, label=f'dt={dt}')

plt.figure(fig_err_osc)
plt.xlabel("time (s)")
plt.ylabel("$E_k$")
fig_err_osc.set_size_inches(w=5, h=3.5)
flnm = os.path.join(os.path.dirname(__file__), f"../figs_linadv/osc_err.pgf")
plt.savefig(flnm)

plt.figure(fig_osc_sim)
plt.subplot(2,1,1)
plt.plot(tref, osc_ref[ex,0,0,:],'k',label='reference')
plt.ylabel("1st component")
plt.subplot(2,1,2)
plt.plot(tref, osc_ref[ex, 1,0,:],'k',label='reference')
plt.legend(fontsize="9")
plt.xlabel("time (s)")
plt.ylabel("2nd component")
fig_osc_sim.set_size_inches(w=5, h=3.5)
flnm = os.path.join(os.path.dirname(__file__), f"../figs_linadv/osc_sim.pgf")
plt.savefig(flnm)

plt.figure(fig_nlin_max)
plt.loglog(dtlist, osc_max_error, 'k', alpha=0.8, label=f"harmonic oscillator")

plt.figure(fig_nlin_mse)
plt.loglog(dtlist, osc_mse_error, 'k', alpha=0.8, label=f"harmonic oscillator")

plt.figure(fig_nlin_max)
plt.title("Van der Pol oscillator, max")
plt.xlabel("time step (s)")
plt.legend()

plt.figure(fig_nlin_mse)
plt.title("Van der Pol oscillator, MSE")
plt.legend()
plt.xlabel("time step (s)")

plt.figure(fig_loss)
plt.xlabel("Training iterations")
plt.ylabel("MSE loss")
plt.legend()

plt.figure(fig_loss_osc)
plt.xlabel("Training iterations")
plt.ylabel("MSE loss")
plt.legend()
fig_loss_osc.set_size_inches(w=5, h=3.5)
flnm = os.path.join(os.path.dirname(__file__), f"../figs_linadv/osc_loss.pgf")
plt.savefig(flnm)


# Spatial grid
m = 16    # Number of grid points in space
L = 1.0   # Width of spatial domain

# Wavenumber "grid"
xi = np.fft.fftfreq(m)*m*2*np.pi/L

uadv = 1.0
nu = 0.0

# define spatial operator matrix
D = -1j*xi*uadv - nu*xi**2
D = np.reshape(D, (1,m))

linadv = problem_linadv(D, uadv, nu)
tend = 20

colors=iter(plt.cm.rainbow(np.linspace(0, 1, len(dtlist))))
linadv_mse_error = []
linadv_max_error = []

init = linadv.get_rand_init(ntests, 1, L)[:, np.newaxis,:]
linadv_init = solution(linadv, init)
linadv_intref = intExact_linadv(0, t_end, nsteps_ref, linadv)
linadv_ref = linadv_intref.run_all(linadv_init).real


for i in range(len(dtlist)):
    dt = dtlist[i]
    rgb = next(colors)
    print(f"working on dt = {dt}")
    nsteps = int(t_end/dt)
    linadv_NN = intNN_linadv(0, t_end, nsteps, linadv)
    linadv_sim = linadv_NN.run_all(linadv_init)
    t = np.arange(0, t_end, dt)
    step_gap_ref = int(dt/dtref)
    linadv_ref_ts = linadv_ref[:,:,:,np.arange(0,nsteps_ref,step_gap_ref)]
    linadv_diff = abs(linadv_ref_ts - linadv_sim)
    linadv_mean = np.mean(linadv_diff, axis=0)
    linadv_max = np.max(linadv_diff, axis=3)
    linadv_mse_error.append(np.mean(np.mean(np.mean(linadv_diff**2, axis=0), axis=2)))
    linadv_max_error.append(np.mean(linadv_max))


plt.show()
