from solution_m import solution

class integrator():
    def __init__(self, tstart, tend, nsteps, problem):
        assert tstart<tend, "tstart must be smaller than tend"
        assert (isinstance(nsteps, int) and nsteps>0), "nsteps must be a positive integer"
        self.tstart = tstart
        self.tend   = tend
        self.nsteps = nsteps
        self.dt     = (tend - tstart)/float(nsteps)
        self.problem= problem

    # Run integrator from tstart to tend using nstep many steps
    def run(self, u0):
        assert isinstance(u0, solution), "Initial value u0 must be an object of type solution"
        raise NotImplementedError("Function run in generic integrator not implemented: needs to be overloaded in derived class")

    def run_all(self, u0):
        """
        returns array of shape nprob x nvar x nspace x nsteps
        """
        assert isinstance(u0, solution), "Initial value u0 must be an object of type solution"
        raise NotImplementedError("Function run_all in generic integrator not implemented: needs to be overloaded in derived class")
