import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))

import numpy as np
from problem_vdp_m import problem_vdp
from solution_m import solution
from parareal_m import parareal
from intRK4_m import intRK4
from intNN_vdp_m import intNN_vdp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#nog iets mis met meerdere parareal tegelijk
# np.random.seed(2)
mu = 4
problem = problem_vdp(mu)

tstart = 0
tend = 16
nslices = 16
# nfine = 20
dtfine = 0.01
nfine = round((tend-tstart)/nslices/dtfine)
# ncoarse = 4
dtcoarse = 0.5
ncoarse = round((tend-tstart)/nslices/dtcoarse)

tol = 1e-6
itmax = nslices
x0 = np.random.uniform(-2, 2, 1)
y0 = np.random.uniform(-2*mu+0.5, 2*mu+0.5,1)
init = np.array((x0, y0))
init = np.reshape(init, (1,2,1))
init_sol = solution(problem, init)
para = parareal(tstart, tend, nslices, intRK4, intNN_vdp, nfine, ncoarse, tol, itmax, init_sol)
para.run()
print(f"final max residual is {para.get_final_res()}")
y = para.get_all_end_values()
y_fine = para.get_all_end_values_serial_fine()
y_coarse = para.get_all_end_values_serial_coarse()

t_fine = np.linspace(tstart, tend, nslices*nfine, endpoint=False)
t_coarse = np.linspace(tstart, tend, nslices*ncoarse, endpoint=False)

ex = 0

fig1 = plt.figure()
plt.subplot(2,1,1)
plt.title('vdp, real')
# plt.plot(t_fine, y[0,0,:], label='parareal')
plt.plot(t_fine, y[ex,0,0,:], label='parareal')
plt.plot(t_fine, y_fine[ex,0,0,:],'--', label='fine')
plt.plot(t_coarse, y_coarse[ex,0,0,:], label='coarse')
plt.plot(0, init[ex,0], '*')
plt.legend()
plt.subplot(2,1,2)
plt.title('vdp, imag')
plt.plot(t_fine, y[ex,1,0,:], label='parareal')
plt.plot(t_fine, y_fine[ex,1,0,:], label='fine')
plt.plot(t_coarse, y_coarse[ex,1,0,:], label='coarse')
plt.legend()
plt.xlabel("time")

plt.figure()
plt.subplot(2,1,1)
plt.title('parareal error van der pol')
plt.semilogy(t_fine, np.abs(y[ex,0,0,:]-y_fine[ex,0,0,:]))
plt.subplot(2,1,2)
plt.semilogy(t_fine, np.abs(y[ex,1,0,:]-y_fine[ex,1,0,:]))
plt.xlabel("time")

# fig, ax = plt.subplots(1, 1)
# y_an = y_coarse
# t_an = t_fine
# dt_an = t_an[1]-t_an[0]
# remax = np.max(y_an[0,:,:])
# remin = np.min(y_an[0,:,:])
# immax = np.max(y_an[1,:,:])
# immin = np.min(y_an[1,:,:])
# ax.set_xlim([remax*1.1, remin*1.1])
# ax.set_ylim([immax*1.1,immin*1.1])
# plt.xlabel('real')
# plt.ylabel('imag')
# intv = 1

# def animate(i):
#     t = i * dt_an * intv
#     ax.plot(y_an[0,0,i*intv], y_an[1,0,i*intv],'.', markersize=1)
#     # ax.plot(x, data[i * intv,1,:],'r', linewidth=1.5)
#     # ax.plot(x, data[i * intv,2,:],'g', linewidth=1.5)
#     plt.title('t={:.2f}'.format(t))

# anim = animation.FuncAnimation(fig, animate, frames = int(np.shape(y_an)[2]/intv),
#      interval = 1, repeat = False)

plt.show()