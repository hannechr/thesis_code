import numpy as np

# Class representing the solution of an IVP with nprob different init values

class solution(object):

  def __init__(self, problem, y):
    # y is solution, form nprob x nvar x nspat
    assert isinstance(y, np.ndarray), "Argument y must be of type numpy.ndarray"
    assert np.ndim(y) == 3
    # assert np.shape(y)[0]==np.size(y), "Argument y must be a linear array"
    # If y is a purely 1D array, reshape it into a Nx1xnprob 3D array... if both types are mixed, horrible inconsistencies arise
    # self.y    = np.reshape(y, (np.shape(y)[0], 1, np.shape(y)[-1]))
    self.y = y
    self.nprob, self.ndim, self.nspat = np.shape(y)
    self.problem = problem
    # assert np.array_equal( np.shape(problem.M), [self.ndof, self.ndof]), "Matrix M does not match size of argument y"

  # Overwrite y with a*x+y
  def axpy(self, a, x):
    assert (np.size(a)==1 or isinstance(a, float)), "Input a must be a scalar"
    # Ask for foregiveness instead of permission...
    try:
      self.y = a*x.y + self.y
    except:
      assert isinstance(x, solution), "Input x must be an object of type solution"
      self.check_dim(x)
      raise Exception('Unknown error in solution.axpy')

  # Overwrite y with f(y)
  def f(self):
    self.problem.f(self)

  # Overwrite y with solution of M*y-alpha*f(y) = y
  def solve(self, alpha):
    self.problem.solve(self, alpha)

  # Return inf norm of y
  def norm(self):
    return np.linalg.norm(self.y, np.inf, axis=(1,2))

  def check_dim(self, x=None):
    if x is None:
      x = self.y
    assert np.ndim(x) == 3
    nprob, ndim, nspat = np.shape(x)
    assert ndim == self.ndim, "Number of dimensions is different in x than in this solution object"
    assert nprob == self.nprob, "Number of problems doesn't match"
    assert nspat == self.nspat, "Spatial resulution doesn't match"