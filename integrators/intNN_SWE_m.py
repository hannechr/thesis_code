from integrator_m import integrator
from solution_m import solution
from net_m import Net
import pickle
import torch
import numpy as np
import os

class intNN_SWE(integrator):

    def __init__(self, tstart, tend, nsteps, problem, data=None):
        super(intNN_SWE, self).__init__(tstart, tend, nsteps, problem)
        ntraindata = 10000
        length_run = 30
        tbet = 3
        crit = 'MSE'
        nbneurons = problem.nbneurons
        curr_file = os.path.dirname(__file__)
        netfile = curr_file + f"/../networks/swe/net_{problem.nspat}_{self.dt:.4f}_{problem.hvar}" + \
                    f"_{problem.havg}_{ntraindata}_{length_run}_{nbneurons}_{crit}_{problem.L}_{tbet}"
        try:
            f = open(netfile, "rb")
            self.net = pickle.load(f)
        except:
            print(f"Neural net for dt={self.dt:.4f} doesn't exist yet. Starting to train one ...")
            model = Net([problem.nspat*(problem.nvar),nbneurons,nbneurons,problem.nspat*(problem.nvar)], self.dt, 1)
            if data is None:
                data = problem.generateData(ntraindata, self.dt, 0.01, length_run=length_run)
            model.train(data, max_epoch=500000, model_path=netfile)
            self.net = model

    def __str__(self):
        return 'NN'
        
    def run(self, u0):
        assert isinstance(u0, solution), "u0 should be a solution instance"
        # uv = u0.y[:,0:2,:]
        x = np.fft.ifft(u0.y, axis=2).real
        x = torch.from_numpy(x)
        shp = x.shape
        x = x.reshape(shp[0], shp[1]*shp[2])
        for i in range(self.nsteps):
            x = self.net.forward(x)
        x = x.reshape(shp)
        x = np.fft.fft(x.detach().cpu().numpy(), axis=2)
        # u0.y = np.concatenate((uv, x[:,2,:]), axis=1)
        u0.y = x

    def run_all(self, u0):
        assert isinstance(u0, solution)
        # uv = u0.y[:,0:2,:]
        all_int = np.zeros((u0.nprob, u0.ndim, u0.nspat, self.nsteps), dtype=np.complex64)
        # x = np.fft.ifft(u0.y[:,[2],:], axis=2).real
        x = np.fft.ifft(u0.y, axis=2).real
        x = torch.from_numpy(x)
        shp = x.shape
        x = x.reshape(shp[0], shp[1]*shp[2])
        for i in range(self.nsteps):
            x_int = np.fft.fft(x.reshape(shp).detach().cpu().numpy(), axis=2)
            # all_int[:,0:2,:,i] = uv
            all_int[:,:,:,i] = x_int
            x = self.net.forward(x)
        return all_int

    def run_last(self, u0):
        assert isinstance(u0, solution)
        x = np.fft.ifft(u0.y, axis=2).real
        x = torch.from_numpy(x)
        shp = x.shape
        x = x.reshape(shp[0], shp[1]*shp[2])
        for _ in range(self.nsteps):
            x = self.net.forward(x)
        return np.fft.fft(x.reshape(shp).detach().cpu().numpy(), axis=2)
