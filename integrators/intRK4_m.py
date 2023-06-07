from integrator_m import integrator
from solution_m import solution
import numpy as np


class intRK4(integrator):

    def __init__(self, tstart, tend, nsteps, problem):
        super(intRK4, self).__init__(tstart, tend, nsteps, problem)

    def run(self, u0):
        assert isinstance(u0, solution), "u0 should be a solution instance"
        y = u0.y
        for _ in range(self.nsteps):
            k1 = self.problem.f_return(y)
            k2 = self.problem.f_return(y + self.dt*k1/2)
            k3 = self.problem.f_return(y + self.dt*k2/2)
            k4 = self.problem.f_return(y + self.dt*k3)
            y = y + 1/6 * self.dt * (k1 + 2*k2 + 2*k3 + k4)
        u0.y = y

    def run_all(self, u0):
        assert isinstance(u0, solution)
        all_int = np.zeros((u0.nprob, u0.ndim, u0.nspat, self.nsteps), dtype=np.complex128)
        y = u0.y
        for i in range(self.nsteps):
            all_int[:,:,:,i] = y
            k1 = self.problem.f_return(y)
            k2 = self.problem.f_return(y + self.dt*k1/2)
            k3 = self.problem.f_return(y + self.dt*k2/2)
            k4 = self.problem.f_return(y + self.dt*k3)
            y = y + 1/6 * self.dt * (k1 + 2*k2 + 2*k3 + k4)
            # if i%(self.nsteps/100) == 0:
            #     print(f"{i/self.nsteps*100:.0f} %", end=" - ")
        return all_int
    
    def run_last(self, u0):
        y = u0.y
        for _ in range(self.nsteps):
            k1 = self.problem.f_return(y)
            k2 = self.problem.f_return(y + self.dt*k1/2)
            k3 = self.problem.f_return(y + self.dt*k2/2)
            k4 = self.problem.f_return(y + self.dt*k3)
            y = y + 1/6 * self.dt * (k1 + 2*k2 + 2*k3 + k4)
        return y