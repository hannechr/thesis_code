import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "integrators"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sol-prob"))

import numpy as np
from parareal_m import parareal
from intExact_oscillator_m import intExact_oscillator
from intBWE_m import intBWE
from intRK4_m import intRK4
from intNN_oscillator_m import intNN_oscillator
from solution_m import solution
from problem_oscillator_m import problem_oscillator
import matplotlib.pyplot as plt

a = 1
problem = problem_oscillator(a)

tstart = 0
tend = 16
nslices = 16
nfine = 10
# ncoarse = 2
dtcoarse = 0.1
ncoarse = round((tend-tstart)/nslices/dtcoarse)
tol = 1e-6
itmax = nslices
r0 = np.random.uniform(0, 1.5, 1)
phi0 = np.random.uniform(-np.pi, np.pi, 1)
init = np.array([r0*np.sin(phi0), r0*np.cos(phi0)])
init = np.reshape(init, (1,2,1))
# solution : nsol x nvar x nspat
init_sol = solution(problem, init)
para = parareal(tstart, tend, nslices, intExact_oscillator, intNN_oscillator, nfine, ncoarse, tol, itmax, init_sol)
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
plt.title('oscillator, real')
plt.plot(t_fine, y[ex,0,0,:], label='parareal')
plt.plot(t_fine, y_fine[ex,0,0,:],'--', label='fine')
plt.plot(t_coarse, y_coarse[ex,0,0,:], label='coarse')
plt.plot(0, init[ex,0], '*')
plt.legend()
plt.subplot(2,1,2)
plt.title('oscillator, imag')
plt.plot(t_fine, y[ex,1,0,:], label='parareal')
plt.plot(t_fine, y_fine[ex,1,0,:], label='fine')
plt.plot(t_coarse, y_coarse[ex,1,0,:], label='coarse')
plt.legend()
plt.xlabel("time")


plt.figure()
plt.subplot(2,1,1)
plt.title('parareal error oscillator')
plt.semilogy(t_fine, np.abs(y[ex,0,0,:]-y_fine[ex,0,0,:]))
plt.subplot(2,1,2)
plt.semilogy(t_fine, np.abs(y[ex,1,0,:]-y_fine[ex,1,0,:]))
plt.xlabel("time")

plt.show()

