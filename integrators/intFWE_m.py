from integrator_m import integrator
from solution_m import solution
import numpy as np

class intFWE(integrator):

    def __init__(self, tstart, tend, nsteps, problem):
        super(intFWE, self).__init__(tstart, tend, nsteps, problem)

    def run(self, u0):
        assert isinstance(u0, solution), "u0 should be a solution instance"
        y = u0.y
        for _ in range(self.nsteps):
            y = y + self.dt*self.problem.f_return(y)
        u0.y = y
            
    def run_all(self, u0):
        assert isinstance(u0, solution), "Exact integrator intexact can only be used for solution_oscillator type initial values"
        all_int = np.zeros((u0.nprob, u0.ndim, u0.nspat, self.nsteps), dtype=np.complex64)
        y = u0.y
        for i in range(self.nsteps):
            all_int[:,:,:,i] = u0.y
            y = y + self.dt*self.problem.f_return(y)
        return all_int
    
    def run_last(self, u0):
        assert isinstance(u0, solution), "Exact integrator intexact can only be used for solution_oscillator type initial values"
        y = u0.y
        for _ in range(self.nsteps):
            y = y + self.dt*self.problem.f_return(y)
        return y