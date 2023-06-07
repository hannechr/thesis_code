from timemesh_m import timemesh
from solution_m import solution
import copy
import matplotlib.pyplot as plt
import numpy as np

class parareal(object):

    def __init__(self, tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max, u0):
        assert isinstance(u0, solution), "Argument u0 must be an object of type solution"
        u0.check_dim()
        self.timemesh = timemesh(tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max, u0.problem)
        self.u0 = u0

    def run(self, plot_all = False, xi=None):
        if plot_all:
            plt.figure(figsize=(5.6,4.2))
            m = self.u0.nspat
            print(xi)
            xi = np.fft.fftshift(xi)
            xi = xi[int(m/2):m]/m

        # Coarse predictor; need deepcopy to keep self.u0 unaltered
        self.timemesh.run_coarse(copy.deepcopy(self.u0))

        it = 0
        itmax = self.timemesh.slices[0].iter_max
        res = np.zeros((self.u0.ndim, itmax))
        while it < itmax:
            it +=1
            print(it)

            # Run fine method
            self.timemesh.update_fine_all() # COULD BE PARALLEL - but isn't parallel now?
            # now all the fine values have been calculated, based on estimate from previous round.

            for i in range(0,self.timemesh.nslices):

                # if not it==1:
                #     self.plot_slice(self.timemesh.get_fine_value(i), fig=fig2)
                #     plt.legend(("start of slice", "coarse simulation", "fine simulation"))
                #     plt.show()
                try:
                    end_old = self.timemesh.slices[i].sol_end
                except:
                    end_old = self.timemesh.slices[i].sol_coarse

                # Compute difference F-G (stored in fine)
                fine = self.timemesh.get_fine_value(i) # fine value calculated in update_fine_all
                coarse = self.timemesh.get_coarse_value(i)
                update = copy.deepcopy(fine)
                update.axpy(-1, coarse)

                # Fetch update value from previous time slice
                if i==0: # moet deze lijn?
                    self.timemesh.set_initial_value(self.u0, slice_nr=0)
                else:
                    self.timemesh.set_initial_value(copy.deepcopy(self.timemesh.get_end_value(i-1)), i)

                # Update coarse value
                self.timemesh.update_coarse(i)

                # Perform correction G_new + F_old - G_old
                end = copy.deepcopy(self.timemesh.get_coarse_value(i))
                end.axpy(1.0, update)

                # Set corrected value as new end value
                self.timemesh.set_end_value(end, i)

                if plot_all and it==1 and i in [3,4,5,6,7]:
                    # fig1 = self.plot_slice(end_old)
                    # self.plot_slice(coarse, fig=fig1)
                    # self.plot_slice(fine, fig=fig1)
                    # self.plot_slice(self.timemesh.get_coarse_value(i), fig=fig1, spc=":")
                    # self.plot_slice(end, fig=fig1, spc='--')
                    # plt.title(f"slice {i}, it {it}")
                    # plt.legend(("end old","coarse old", "fine old","coarse_new","end new"))
                    # # plt.show()
                    prevendspec = (self.timemesh.get_end_value(i-1)).y[0,0,:]
                    prevendgrid = np.fft.ifft(prevendspec, axis=-1).real
                    m = np.size(prevendspec)
                    x = np.linspace(0,self.u0.problem.L, m, endpoint=False)
                    # print(prevendgrid)
                    plt.subplot(2,5,i-2)
                    plt.plot(x, prevendgrid, label='$U^1_{p-1}$')
                    coarsespec = self.timemesh.get_coarse_value(i).y[0,0,:]
                    coarsegrid = np.fft.ifft(coarsespec, axis=-1).real
                    plt.plot(x, coarsegrid, label='$\mathcal{G}(U^1_{p-1}$)')
                    plt.xticks([0, 1], fontsize=8)
                    plt.title(f"slice {i}", fontsize=9)
                    if i != 3:
                        plt.yticks([])
                    else:
                        plt.yticks([0.8, 1], fontsize=8)
                        plt.ylabel('$u$', fontsize=9)
                    # if i==7:
                    #     plt.legend(fontsize=7, markerscale=0.1, labelspacing=0.2)
                    plt.ylim([0.6, 1.1])
                    plt.subplot(2,5,5+i-2)
                    plt.semilogy(xi,abs(np.fft.fftshift(prevendspec)[int(m/2):m])/m, label='start of slice')
                    plt.semilogy(xi,abs(np.fft.fftshift(coarsespec)[int(m/2):m])/m, label='coarse')
                    plt.ylim([1e-5, 1e0])
                    plt.yticks([1e-5, 1e-3, 1e-1], fontsize=8)
                    plt.xticks([0, 2.5], fontsize=8)
                    if i != 3:
                        plt.yticks([])
                    else:
                        plt.ylabel('$|\hat{u}|$', fontsize=9)
                    # if i==5:
                        # plt.xlabel("wave number")

                
            # increase iteration counter
            self.timemesh.increase_iter_all() 

            res[:,it-1] = self.get_final_res()

            # stop loop if all slices have converged or itermax reached
            if self.timemesh.all_converged():
                print(f"converged in {it} iterations")
                return it, res
        return it, res

    # return end value of time slice i
    def get_end_value(self, i):
        return self.timemesh.get_end_value(i)
    
    def get_coarse_value(self, i):
        return self.timemesh.get_coarse_value(i)
    
    def get_fine_value(self, i):
        return self.timemesh.get_fine_value(i)
    
    # return end value of last time slice
    def get_last_end_value(self):
        return self.get_end_value(self.timemesh.nslices-1)
    
    def get_last_coarse_value(self):
        return self.get_coarse_value(self.timemesh.nslices-1)
    
    def get_last_fine_value(self):
        return self.get_fine_value(self.timemesh.nslices-1)

    def get_all_end_values(self):
        return self.timemesh.get_all_end_values()
    
    def get_all_end_values_serial_fine(self):
        timemesh_only_fine = copy.deepcopy(self.timemesh)
        timemesh_only_fine.run_fine(self.u0)
        return timemesh_only_fine.get_all_fine_values()
    
    def get_all_end_values_serial_coarse(self):
        timemesh_only_coarse = copy.deepcopy(self.timemesh)
        timemesh_only_coarse.run_coarse(self.u0)
        return timemesh_only_coarse.get_all_coarse_values()


    def get_final_res(self):
        return self.timemesh.get_max_residual()
    

    def plot_slice(self, sol, fig=None, spc='-'):
        if fig is None:
            fig = plt.figure(figsize=(2,2))
        plt.figure(fig)
        nvar = sol.ndim
        y = np.fft.ifft(sol.y[0,:,:], axis=-1).real
        for i in range(nvar):
            plt.subplot(nvar,1,i+1)
            plt.plot(y[i,:], spc)
        return fig