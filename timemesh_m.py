import numpy as np
from timeslice_m import timeslice
import copy

class timemesh():

    def __init__(self, tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max, problem):
        assert tstart<tend, "tstart has to be smaller than tend"
        #
        # For the time being, all timeslices are created equal...
        #
        self.timemesh = np.linspace(tstart, tend, nslices+1)
        self.nslices  = nslices
        self.slices   = []
        self.tstart   = tstart
        self.tend     = tend
        self.nsteps_fine = nsteps_fine
        self.nsteps_coarse = nsteps_coarse

        for i in range(0, nslices):
            ts_fine   =   fine(self.timemesh[i], self.timemesh[i+1], nsteps_fine, problem)
            ts_coarse = coarse(self.timemesh[i], self.timemesh[i+1], nsteps_coarse, problem)
        
            self.slices.append(timeslice(ts_fine, ts_coarse, tolerance, iter_max))


          # Run the coarse method serially over all slices

    # Run the coarse method serially over all slices
    def run_coarse(self, u0):
        self.set_initial_value(u0, slice_nr=0)
        for i in range(0,self.nslices):
        # Run coarse method
            self.slices[i].update_coarse()
            # Fetch coarse value and set initial value of next slice
            if i<self.nslices-1:
                self.set_initial_value(copy.deepcopy(self.get_coarse_value(i)), i+1)

    # Run the fine method serially over all slices
    def run_fine(self, u0):
        self.set_initial_value(u0, slice_nr=0)
        for i in range(0,self.nslices):
        # Run fine method
            self.slices[i].update_fine()
            # Fetch fine value and set initial value of next slice
            if i<self.nslices-1:
                self.set_initial_value(copy.deepcopy(self.get_fine_value(i)), i+1)

    # Update fine values for all slices
    # @NOTE: This is not equivalent to run_fine, since no updated initial values are copied forward, thus this works parallel
    def update_fine_all(self):
        for i in range(0,self.nslices):
            self.slices[i].update_fine()

    # Update coarse values for all slices
    # @NOTE: This is not equivalent to run_fine, since no updated initial values are copied forward
    def update_coarse_all(self):
        for i in range(0,self.nslices):
            self.slices[i].update_coarse()

    def update_coarse(self, i):
        self.slices[i].update_coarse()

    def update_fine(self, i):
        self.slices[i].update_fine()

      # increase iteration counter of a single time slice
  
    def increase_iter(self, slice_nr):
        assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
        self.slices[slice_nr].increase_iter()

    # increase iteration counters of ALL time slices
    def increase_iter_all(self):
        for i in range(0,self.nslices):
            self.increase_iter(i)


    # ===== SET functions ===== #

    def set_initial_value(self, u0, slice_nr=0):
        assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
        self.slices[slice_nr].set_sol_start(u0)

    def set_end_value(self, u0, slice_nr=0):
        assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
        self.slices[slice_nr].set_sol_end(u0)


    # ===== GET functions ===== #

    # returns value at end of slice, using coarse
    def get_coarse_value(self, slice_nr):
        assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
        return self.slices[slice_nr].get_sol_coarse()

    def get_coarse_integr(self, slice_nr):
        assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
        return self.slices[slice_nr].get_sol_coarse_integr()

    # returns value at end of slice, using fine
    def get_fine_value(self, slice_nr):
        assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
        return self.slices[slice_nr].get_sol_fine()
    
    def get_fine_integr(self, slice_nr):
        assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
        return self.slices[slice_nr].get_sol_fine_integr()
    
    # returns value at end of slice, using combi fine/coarse
    def get_end_value(self, slice_nr):
        assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
        return self.slices[slice_nr].get_sol_end()
    
    def get_end_integr(self, slice_nr):
        assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
        return self.slices[slice_nr].get_sol_end_integr()

    def get_all_end_values(self):
        sol0 = self.slices[0].get_sol_end()
        u_end = np.zeros((sol0.nprob, sol0.ndim, sol0.nspat, self.nslices*self.nsteps_fine), dtype=np.complex64)
        for i in range(0, self.nslices):
            u_end[:,:,:,i*self.nsteps_fine:(i+1)*self.nsteps_fine] = self.get_end_integr(i)
        return u_end
    
    def get_all_fine_values(self):
        sol0 = self.slices[0].get_sol_fine()
        u_fine = np.zeros((sol0.nprob, sol0.ndim, sol0.nspat, self.nslices*self.nsteps_fine), dtype=np.complex64)
        for i in range(0, self.nslices):
            u_fine[:,:,:,i*self.nsteps_fine:(i+1)*self.nsteps_fine] = self.get_fine_integr(i)
        return u_fine
    
    def get_all_coarse_values(self):
        sol0 = self.slices[0].get_sol_coarse()
        u_coarse = np.zeros((sol0.nprob, sol0.ndim, sol0.nspat, self.nslices*self.nsteps_coarse), dtype=np.complex64)
        for i in range(0, self.nslices):
            u_coarse[:,:,:,i*self.nsteps_coarse:(i+1)*self.nsteps_coarse] = self.get_coarse_integr(i)
        return u_coarse

    def get_max_residual(self):
        maxres = self.slices[0].get_residual()
        for i in range(1,self.nslices):
            # print(f"residual slice {i}: {self.slices[i].get_residual()}")
            maxres = np.maximum( maxres, self.slices[i].get_residual())
        return maxres

  # ===== BOOLEAN functions ===== #

    def all_converged(self):
        all_converged = True
        i = 0
        while (all_converged and i<self.nslices):
            if not self.slices[i].is_converged():
                all_converged = False
            i += 1
        return all_converged
    