from integrator_m import integrator
from solution_m import solution
from net_m import Net
import pickle
import torch
import numpy as np
import os

class intNN_vdp(integrator):

    def __init__(self, tstart, tend, nsteps, problem, data=None, smart=True):
        super(intNN_vdp, self).__init__(tstart, tend, nsteps, problem)
        curr_file = os.path.dirname(__file__)
        netfile = curr_file + f"/../networks/vdp/Net_mu_{problem.mu}_{self.dt:.4f}_{smart}"
        try:
            f = open(netfile, "rb")
            self.net = pickle.load(f)
        except:
            print(f"Neural net for dt={self.dt:.4f} doesnt exist yet. Starting to train one ...")
            model = Net([2,16,16,2], self.dt, 1)
            if data is None:
                dtdata = 0.0001
                data = problem.generateData(1000, self.dt, dtdata, smart)
            model.train(data, max_epoch=150000, model_path=netfile)
            self.net = model
        
    def run(self, u0):
        assert isinstance(u0, solution), "u0 should be a solution instance"
        x = torch.from_numpy(u0.y[:,:,0])
        for i in range(self.nsteps):
            x = self.net.forward(x)
        u0.y = x.detach().cpu().numpy()[:,:,np.newaxis]

    def run_all(self, u0):
        assert isinstance(u0, solution)
        all_int = np.zeros((u0.nprob, u0.ndim, u0.nspat, self.nsteps))
        x = torch.from_numpy(u0.y[:,:,0]) # if also spat, reshape
        for i in range(self.nsteps):
            all_int[:,:,0,i] = x.detach().cpu().numpy()
            x = self.net.forward(x)
        return all_int

    def run_last(self, u0):
        assert isinstance(u0, solution)
        x = torch.from_numpy(u0.y[:,:,0]) # if also spat, reshape
        for _ in range(self.nsteps):
            x = self.net.forward(x)
        return x.detach().cpu().numpy()
