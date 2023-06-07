from integrator_m import integrator
from solution_m import solution
import numpy as np

class intExact_oscillator(integrator):

    def __init__(self, tstart, tend, nsteps, problem):
        super(intExact_oscillator, self).__init__(tstart, tend, nsteps, problem)
        a = problem.a
        self.int = np.array([[np.cos(a*self.dt), np.sin(a*self.dt)],[-np.sin(a*self.dt), np.cos(a*self.dt)]])

    def run(self, u0):
        assert isinstance(u0, solution), "Exact integrator intexact can only be used for solution_oscillator type initial values"
        for _ in range(self.nsteps):
            u0.y = np.matmul(self.int, u0.y)
    
    def run_all(self, u0):
        assert isinstance(u0, solution), "Exact integrator intexact can only be used for solution_oscillator type initial values"
        all_int = np.zeros((u0.nprob, u0.ndim, u0.nspat, self.nsteps))
        y = u0.y
        for i in range(self.nsteps):
            all_int[:,:,:,i] = y
            y = np.matmul(self.int, y)
        return all_int

    def run_last(self, u0):
        assert isinstance(u0, solution)
        y = u0.y
        for _ in range(self.nsteps):
            y = np.matmul(self.int, y)
        return y