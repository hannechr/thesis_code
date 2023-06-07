from problem_m import problem
import numpy as np
from scipy.sparse import linalg
from scipy import sparse
import torch

class problem_oscillator(problem):

    def __init__(self, a):
        super(problem_oscillator, self).__init__()
        self.M = sparse.eye(2, format="csc")
        self.a = a
        self.A = np.array([[0,a],[-a,0]])

    # y' = a*1j*y // split Re and Im
    # in vb wordt sparse gebruikt
    def f(self, sol):
        sol.y = np.matmul(self.A,sol.y)

    def f_return(self, y):
        assert np.shape(self.M)[0] == np.shape(y)[0], "input doesn't have correct dimensions"
        return np.matmul(self.A, y)

    def solve(self, sol, alpha):
        # print(type(self.M), type(self.A))
        sol.y = linalg.spsolve(self.M - alpha*self.A, sol.y)
        sol.y = np.reshape(sol.y, (sol.nprob, sol.ndim, sol.nspat))

    def generateData(self, n_sets, dt):
        # shape: n_sets, n_timesteps, n_var
        data = np.zeros((n_sets, 2, 2), dtype=np.double)
        intg = np.array([[np.cos(self.a*dt), np.sin(self.a*dt)],[-np.sin(self.a*dt), np.cos(self.a*dt)]])
        for i in range(n_sets):
            r0 = np.random.uniform(1.0, 2.5)
            phi0 = np.random.uniform(-np.pi, np.pi)
            data[i,0,:] = np.array((r0*np.sin(phi0),r0*np.cos(phi0)))
            data[i,1,:] = np.dot(intg, data[i,0,:])
        return torch.from_numpy(data)
    