from integrator_m import integrator
from solution_m import solution
from net_m import Net
import pickle
import torch
import numpy as np
import os

class intNN_linadv(integrator):

    def __init__(self, tstart, tend, nsteps, problem, data=None):
        super(intNN_linadv, self).__init__(tstart, tend, nsteps, problem)
        ntraindata = 1000
        length_run = 10
        crit = 'MSE'
        nbneurons = problem.nbneurons
        curr_file = os.path.dirname(__file__)
        netfile = curr_file + f"/../networks/linadv/Net_u_{problem.uadv}_nu_{problem.nu}_nsp_{problem.nspat}" + \
                    f"_L_{problem.L}_{ntraindata}_{length_run}_{nbneurons}_{crit}_{self.dt:.4f}"
        try:
            f = open(netfile, "rb")
            self.net = pickle.load(f)
        except:
            print(f"Neural net for dt={self.dt:.4f} doesnt exist yet. Starting to train one ...")
            model = Net([problem.nspat*problem.nvar,nbneurons,nbneurons,problem.nspat*problem.nvar], self.dt, 1)
            if data is None:
                data = problem.generateData(ntraindata, self.dt, length_run=length_run)
            if crit == 'rel':
                criterium = model.my_loss
            else:
                criterium = torch.nn.MSELoss(reduction='none')
            model.train(data, max_epoch=500000, model_path=netfile, crit=criterium)
            self.net = model
        
    def run(self, u0):
        assert isinstance(u0, solution), "u0 should be a solution instance"
        x = np.fft.ifft(u0.y, axis=2).real
        x = torch.from_numpy(x)
        shp = x.shape
        x.reshape(shp[0], shp[1]*shp[2])
        for i in range(self.nsteps):
            x = self.net.forward(x)
        x.reshape(shp)
        u0.y = np.fft.fft(x.detach().cpu().numpy(), axis=2)

    def run_all(self, u0):
        assert isinstance(u0, solution)
        all_int = np.zeros((u0.nprob, u0.ndim, u0.nspat, self.nsteps), dtype=np.complex64)
        x = np.fft.ifft(u0.y, axis=2).real
        x = torch.from_numpy(x)
        shp = x.shape
        x.reshape(shp[0], shp[1]*shp[2])
        for i in range(self.nsteps):
            all_int[:,:,:,i] = np.fft.fft(x.reshape(shp).detach().cpu().numpy(), axis=2)
            x = self.net.forward(x)
        return all_int

    def run_last(self, u0):
        assert isinstance(u0, solution)
        x = np.fft.ifft(u0.y, axis=2).real
        x = torch.from_numpy(x)
        shp = x.shape
        x.reshape(shp[0], shp[1]*shp[2])
        for _ in range(self.nsteps):
            x = self.net.forward(x)
        return np.fft.fft(x.reshape(shp).detach().cpu().numpy(), axis=2)