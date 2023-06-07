from integrator_m import integrator
from solution_m import solution
import numpy as np

class intBWE(integrator):

    def __init__(self, tstart, tend, nsteps, problem):
        super(intBWE, self).__init__(tstart, tend, nsteps, problem)

    def run(self, u0):
        assert isinstance(u0, solution), "u0 should be a solution instance"
        for _ in range(self.nsteps):
            u0.solve(self.dt)
            
    def run_all(self, u0):
        assert isinstance(u0, solution), "Exact integrator intexact can only be used for solution_oscillator type initial values"
        all_int = np.zeros((u0.nprob, u0.ndim, u0.nspat, self.nsteps), dtype=np.complex64)
        for i in range(self.nsteps):
            all_int[:,:,:,i] = u0.y
            u0.solve(self.dt)
        return all_int
    
    def run_last(self, u0):
        assert isinstance(u0, solution), "Exact integrator intexact can only be used for solution_oscillator type initial values"
        for _ in range(self.nsteps):
            u0.solve(self.dt)
        return u0.y