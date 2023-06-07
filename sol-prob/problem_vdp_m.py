from problem_m import problem
import numpy as np
from scipy import sparse
import torch
from intRK4_m import intRK4

class problem_vdp(problem):

    def __init__(self, mu):
        super(problem_vdp, self).__init__()
        self.M = np.eye(2)
        self.mu = mu

    def f(self, sol):
        x = sol.y[:,0,:]
        y = sol.y[:,1,:]
        sol.y = np.stack((y, self.mu*(1-np.power(x,2))*y - x), axis=1)

    def f_return(self, y):
        assert np.ndim(y) == 3, "input doesn't have correct dimensions"
        xx = y[:,0,:]
        yy = y[:,1,:]
        return np.stack((yy, self.mu*(1-np.power(xx,2))*yy - xx), axis=1)
    
    def solve(self, sol, alpha):
        raise NotImplementedError("solve not implemented for nonlinear")
    
    def generateData(self, nb_data, dt, dt_data, smart=False):
        """
        generate data in [-2.5,2.5],[-2*mu, 2*mu]
        output shape: torch nb_data*2*nb_var (2=in/out)
        """
        assert(dt/dt_data % 1 == 0), "dt_data should be a divisor of dt"
        stepgap = int(dt/dt_data)
        intg = intRK4(0, dt, stepgap, self)
    
        if not smart:
            r0 = np.random.uniform(1.0, 2.5, (nb_data, 1, 1))
            phi0 = np.random.uniform(-np.pi, np.pi, (nb_data, 1, 1))
            init = np.concatenate((r0*np.sin(phi0),r0*np.cos(phi0)), axis=1)
            data = np.zeros((nb_data, 2, 2), dtype=np.double)
            data[:,0,:] = init[:,:,0]
            data[:,1,:] = intg.run_last(init)[:,:,0]
        else:
            length_run = 10
            t_between = 0.5/self.mu
            int_between = int(t_between/dt)
            nb_runs = int(nb_data/length_run)
            r0 = np.random.uniform(1.0, 2.5, (nb_runs, 1, 1))
            phi0 = np.random.uniform(-np.pi, np.pi, (nb_runs, 1, 1))
            init = np.concatenate((r0*np.sin(phi0),r0*np.cos(phi0)), axis=1)
            data = np.zeros((nb_runs*length_run, 2, 2), dtype=np.double)
            for i in range(length_run):
                data[nb_runs*i:nb_runs*(i+1),0,:] = init[:,:,0]
                init = intg.run_last(init)
                data[nb_runs*i:nb_runs*(i+1),1,:] = init[:,:,0]
                for _ in range(int_between):
                    init = intg.run_last(init)
        
        return torch.from_numpy(data)

