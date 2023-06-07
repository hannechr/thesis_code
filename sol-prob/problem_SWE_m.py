from problem_m import problem
from solution_m import solution
import numpy as np
from intRK4_m import intRK4
import torch
import matplotlib.pyplot as plt

class problem_SWE(problem):

    def __init__(self, Alin, fnlin, havg, hvar, L, nbneurons=32):
        super(problem_SWE, self).__init__()
        self.Alin = np.moveaxis(Alin,2,0)    # linear component in matrix form
        self.fnlin = fnlin  # nonlinear component in function form
        self.nspat = np.shape(Alin)[2]
        self.nvar = 3
        self.havg = havg
        self.hvar = hvar
        self.L = L
        self.nbneurons = nbneurons

    def f(self, sol):
        sol.y = np.swapaxes(np.matmul(self.Alin,np.swapaxes(sol.y, 0, 2)),0,2) + self.fnlin(sol.y)
        sol.check_dim()

    def f_return(self, y):
        return np.swapaxes(np.matmul(self.Alin,np.swapaxes(y, 0, 2)),0,2) + self.fnlin(y)

    def solve(self, sol, alpha):
        raise NotImplementedError("solve function not possible for nonlinear systems")

    # def SWEloss(self, ytrue, ypred):


    def generateData(self, n_sets, dt, dt_data, length_run=10):
        # shape output: n_sets, 2, ndof
        assert(dt/dt_data % 1 == 0), "dt_data should be a divisor of dt"
        stepgap = int(dt/dt_data)
        intg = intRK4(0, dt, stepgap, self)

        t_between = 3 #5*dt 
        int_between = int(t_between/dt)
        nb_runs = int(n_sets/length_run)
        init = self.get_rand_init(nb_runs)
        data = np.zeros((nb_runs*length_run, 2, self.nspat*self.nvar), dtype=np.double)
        progress=0
        print('--')
        for i in range(length_run):
            data[nb_runs*i:nb_runs*(i+1),0,:] = np.reshape(init, (nb_runs, (self.nvar)*self.nspat))
            inithat = np.fft.fft(init, axis=2)
            inithat = intg.run_last(solution(self, inithat))
            # inithat[:,1,:] = 0
            data[nb_runs*i:nb_runs*(i+1),1,:] = np.reshape(np.fft.ifft(inithat, axis=2).real, (nb_runs, (self.nvar)*self.nspat))
            init = np.fft.ifft(inithat, axis=2).real
            for _ in range(int_between):
                inithat = intg.run_last(solution(self, inithat))
            init = np.fft.ifft(inithat, axis=2).real
            progress += 100/length_run
            print(f"{progress:.2f} %",end="\r")
            
        return torch.from_numpy(data)
    
    # sum[a_n*cos(n*2*pi*x/L + p_n)] // physical space
    def get_rand_init(self, nsets):
        """
        Generate nsets random initial sets, with domain length L and amplitude of first mode=ampl
        shape: nsets x nvar x nspat
        """
        x = np.linspace(0, self.L, self.nspat, endpoint=False, dtype=np.double)
        n = np.min((int(self.nspat/2), 16)) # half of the nb points bcs Nyquist, max 16
        phase_speed = 2*np.pi*np.repeat((np.outer(np.arange(n), x/self.L))[np.newaxis,:,:], nsets, axis=0) # shape: nsets x 16 x nspat
        phase_shift = np.repeat(np.random.uniform(0, 2*np.pi, size=(nsets, n))[:,:,np.newaxis], self.nspat, axis=2) # shape: nsets x 16 x nspat
        arg = phase_speed + phase_shift # matrix containing argument of cos
        coeff = np.random.normal(1,0.2,n)/np.power(2, np.arange(n)) # decreasing coefficient 2**(-n)*(gaussian mu=1, sig=0.3)
        h0 = np.matmul(coeff,np.cos(arg))[:,np.newaxis,:]
        u0 = np.ones_like(h0)*0
        v0 = np.ones_like(h0)*0
        return np.concatenate((u0, v0, h0*self.hvar), axis=1)