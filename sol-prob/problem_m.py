from scipy.sparse import linalg
from scipy import sparse

class problem():
    def __init__(self):
        pass

    # Overwrite y with f(y)
    def f(self, sol):
        raise NotImplementedError("Function f in generic solution not implemented: needs to be overloaded in derived class")

    def f_return(self, y):
        raise NotImplementedError("Function f in generic solution not implemented: needs to be overloaded in derived class")

    def solve(self, sol, alpha):
        raise NotImplementedError("Function solve in generic solution not implemented: needs to be overloaded in derived class")
    
    def generate_data(self, sol, alpha):
        raise NotImplementedError("Function generate_data in generic solution not implemented: needs to be overloaded in derived class")
    
    def applyM(self, sol):
        raise NotImplementedError("Function generate_data in generic solution not implemented: needs to be overloaded in derived class")