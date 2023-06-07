import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))
import numpy as np
from problem_SWE_m import problem_SWE
from solution_m import solution
from intRK4_m import intRK4
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

figs = (5.6,3.6)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

np.random.seed(1234)

# Spatial grid
m = 64         # Number of grid points in space
L = 30000      # Width of spatial domain

xi = np.fft.fftfreq(m)*m*2*np.pi/L

# Temporal grid
tstart = 0
tend = 200
nsteps = 5000
dt = 0.05
t = np.linspace(tstart, tend, nsteps, endpoint=False)

# Physical variables
h_avg = 1000        # Average fluid heigth
h_ampl = 10        # Order of wave amplitudes
nu = 0#4e8           # Hyperviscosity coefficient -> groter voor hogere m
ord_hyp = 4         # Order of the hyperviscosity
theta = 0.01        # Angle at which simulation takes place
g = 9.81            # Gravitation
w_earth = 7.29e-5   # Earth rotational speed
f = 2*w_earth*np.sin(theta) # Coriolis effect

print(max(xi))
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

problem = problem_SWE(D, nlin, h_avg, h_ampl, L, nbneurons = 32)

lambda_max = np.sqrt(max(xi**2)*g*h_avg+f**2)
lambda_min = np.sqrt(min(xi**2)*g*h_avg+f**2)

assert lambda_max*dt < 2.8, "Did not start execution because dt is too big for stability"

integrator = intRK4(tstart, tend, nsteps, problem)

# Initial data
x  = np.linspace(0, L, m, endpoint=False)
w0 =  problem.get_rand_init(1)
what0 = np.fft.fft(w0, axis=2)
what0_sol = solution(problem, what0)

whatall = integrator.run_all(what0_sol)
wall = np.fft.ifft(whatall, axis=2).real

# time = np.arange(0,0.31,0.1)  # h=1 -> 0.32   h=2 -> 3.84 
time = np.array([0,100, 200])
indices = np.round(time/dt).astype(int)
# indices = np.round(np.array([0.09*nsteps, 0.095*nsteps, 0.1*nsteps])).astype(int)


testnb = 0
fig = plt.figure(figsize=figs)
gs = fig.add_gridspec(3, hspace=0.15)
axs = gs.subplots(sharex=True)
for index in indices:
    axs[0].plot(x, wall[testnb,0,:,index], label=f't={index*dt:.2f}')
    axs[1].plot(x, wall[testnb,1,:,index], label=f't={index*dt:.2f}')
    axs[2].plot(x, wall[testnb,2,:,index], label=f't={index*dt:.2f}')

plt.subplot(3,1,1)
plt.ylabel("$u$ [m/s]")
plt.legend(loc='lower right', ncol=3, fontsize=7)
plt.subplot(3,1,2)
# plt.yticks([-0.00005, 0, 0.00])
plt.ylabel("$v$ [m/s]")
plt.subplot(3,1,3)
plt.ylabel("$h'$ [m]")
plt.xlabel("$x$ [m]")

plt.subplots_adjust(left=0.15)

plt.savefig('high-h.pgf')


# intv=20
# fig = plt.figure()

# def animate(i):
#     plt.cla()
#     t = i * dt * intv
#     plt.plot(x,wall[testnb,2,:,i*intv],'k--', linewidth=1.5)

#     plt.ylim([-h_ampl, h_ampl])
#     plt.xlabel('x [m]')
#     plt.ylabel('')
#     plt.title('t={:.2f}'.format(t))

# anim = animation.FuncAnimation(fig, animate, frames = int(np.shape(wall)[-1]/intv),
#      interval = 10, repeat = False)



# plt.show()
