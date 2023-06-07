import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))
import numpy as np
from problem_SWE_m import problem_SWE
from intNN_SWE_m import intNN_SWE
from solution_m import solution
from intRK4_m import intRK4
import matplotlib
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

half_figs = (5.6,2.6)
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
dtlist =  np.array((0.100, 0.2500, 1, 2.5, 10, 25))
t_end = 250
ntests = 128
dtref = 0.05
nsteps_ref = int(t_end/dtref)

# Spatial grid
m = 16     # Number of grid points in space
L = 30000      # Width of spatial domain

# Wavenumber "grid"
xi = np.fft.fftfreq(m)*m*2*np.pi/L

# Physical variables
h_avg = 1000         # Average fluid heigth
h_ampl = 100          # Order of wave amplitudes
nu = 0#1e10              # Hyperviscosity coefficient
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

lambda_max = np.sqrt(max(xi**2)*g*h_avg+f**2)
lambda_min = np.sqrt(min(xi**2)*g*h_avg+f**2)

print(lambda_max*dtref, 2.8)
assert lambda_max*dtref < 2.8, "Did not start execution because dtref is too big for stability"

problem = problem_SWE(D, nlin, h_avg, h_ampl, L, nbneurons = 64)

u0    = problem.get_rand_init(ntests) # ntests*nvar*nspat
u0hat = np.fft.fft(u0, axis=2)

init = solution(problem, u0hat)
intref = intRK4(0, t_end, nsteps_ref, problem)
print("working on reference solution")

# try:
#     with open('SWEref.npy', 'rb') as f:
#         refhat = np.load(f)
# except:
refhat = intref.run_all(init)
#     with open('SWEref.npy', 'wb') as f:
#         np.save(f, refhat)

error = []
simhat_list = []
Em_list = []
Ep_list = []
intNN_list = []

for i in range(len(dtlist)):
    dt = dtlist[i]
    print(f"working on dt = {dt}")
    nsteps = int(t_end/dt)
    intNN = intNN_SWE(0, t_end, nsteps, problem)
    intNN_list.append(intNN)
    simhat = intNN.run_all(init)
    simhat_list.append(simhat)
    step_gap_ref = int(dt/dtref)
    refhat_ts = refhat[:,:,:,np.arange(0,nsteps_ref,step_gap_ref)]
    ref_ts = np.fft.ifft(refhat_ts, axis=2)
    sim = np.fft.ifft(simhat, axis=2)
    Epm = np.max(abs(ref_ts - sim), axis=(1,2))
    Em = np.max(Epm, axis=1)
    Ep = np.mean(Epm, axis=0)
    Ep_list.append(Ep)
    E = np.mean(Em)
    error.append(E)

plt.figure(figsize=half_figs)
colors = list(mcolors.TABLEAU_COLORS.keys())

for i in [0,1,2,3,4,5]:
    col = colors[i]
    dt = dtlist[i]
    t = np.arange(0, t_end, dt)
    plt.loglog(t[1:], Ep_list[i][1:], color=col, label=f"$\Delta t={dtlist[i]}$")

plt.loglog([2e-1, 2e2],[5e-3, 5e0], 'k--', label="$\mathcal{O}(N)$")
plt.legend(fontsize=fs_legend, ncol=2)
plt.ylim(bottom=1e-2)
plt.xlabel("$t$[s]")
plt.ylabel("$E_p$")
plt.tight_layout()
plt.savefig("SWEerrorNN.pgf")

# dtshow = 10.0
# nshow = int(t_end/dtshow)
# dtfine = dtref
# intv = 1
# fcratio = int(dtshow/dtref)
# x = np.linspace(0,L, m, endpoint=False)

# initan = u0hat[[0],:,:]
# initann = solution(problem, initan)
# intNN = intNN_SWE(0, t_end, nshow, problem)
# sol = intNN.run_all(initann)

# y_fine = np.fft.ifft(refhat, axis=2)[0,2,:,:].real
# y_coarse = np.fft.ifft(sol, axis=2)[0,2,:,:].real

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
#      interval = 10, repeat = False)


plt.show()