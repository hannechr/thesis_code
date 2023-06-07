from problem_m import problem
from intExact_linadv_m import intExact_linadv
from solution_m import solution
import numpy as np
from scipy.sparse import linalg
from scipy import sparse
import torch

class problem_linadv(problem):

    def __init__(self, A, uadv, nu, L, nbneurons=32):
        super(problem_linadv, self).__init__()
        self.A = A
        self.nspat = np.shape(A)[1]
        self.nvar = 1
        self.uadv = uadv
        self.nu = nu
        self.L = L
        self.nbneurons = nbneurons

    # y' = A*y = f(y) IN SPECTRAL SPACE
    def f(self, sol): 
        # y: nprob x nvar x nspat
        # A: nvar x nspat
        sol.y = self.A*sol.y
        sol.check_dim()

    def f_return(self, y):
        return self.A*y

    def solve(self, sol, alpha):
        sol.y = sol.y/(1-alpha*np.repeat(self.A[np.newaxis,:,:],sol.nprob, axis=0))

    def generateData(self, nb_data, dt, length_run=10):
        # shape output: n_sets, 2, ndof (=nvar x nspat) IN PHYSICAL SPACE!
        intg = intExact_linadv(0, dt, 1, self)
        t_between = 0.8
        nb_runs = int(nb_data/length_run)
        int_between = int(t_between/dt)
        init = self.get_rand_init(nb_runs, L=self.L)
        data = np.zeros((nb_runs*length_run, 2, self.nspat*self.nvar), dtype=np.double)
        for i in range(length_run):
            data[nb_runs*i:nb_runs*(i+1),0,:] = np.reshape(init, (nb_runs, (self.nvar)*self.nspat))
            inithat = np.fft.fft(init, axis=2)
            inithat = intg.run_last(solution(self,inithat))
            # inithat[:,1,:] = 0
            data[nb_runs*i:nb_runs*(i+1),1,:] = np.reshape(np.fft.ifft(inithat, axis=2).real, (nb_runs, (self.nvar)*self.nspat))
            init = np.fft.ifft(inithat, axis=2).real
            for _ in range(int_between):
                inithat = intg.run_last(solution(self,inithat))
            init = np.fft.ifft(inithat, axis=2).real
        return torch.from_numpy(data)
    
    # sum[a_n*cos(n*2*pi*x/L + p_n)] // physical space
    def get_rand_init(self, nsets, ampl=1.0, L=1.0):
        """
        Generate nsets random initial sets, with domain length L and amplitude of first mode=ampl
        shape: nsets x nvar x nspat
        """
        x = np.linspace(0, L, self.nspat, endpoint=False, dtype=np.double)
        n = np.min((int(self.nspat/2), 16)) # half of the nb points bcs Nyquist, max 16
        phase_speed = 2*np.pi*np.repeat((np.outer(np.arange(n), x/L))[np.newaxis,:,:], nsets, axis=0) # shape: nsets x 16 x nspat
        phase_shift = np.repeat(np.random.uniform(0, 2*np.pi, size=(nsets, n))[:,:,np.newaxis], self.nspat, axis=2) # shape: nsets x 16 x nspat
        arg = phase_speed + phase_shift # matrix containing argument of cos
        coeff = np.random.normal(1,0.2,n)*ampl/np.power(2, np.arange(n)) # decreasing coefficient 2**(-n)*(gaussian mu=1, sig=0.3)
        return np.matmul(coeff,np.cos(arg))[:,np.newaxis,:]