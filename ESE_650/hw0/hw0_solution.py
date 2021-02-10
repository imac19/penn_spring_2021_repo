import torch
import numpy as np

class HW0Solution:
    def __init__(self):
        pass

    # Simulate the system Ax+b for one timestep using the given initial conditions
    # A = NxN numpy array
    # b = 1D N numpy array
    # init_cond = N x 1 x M numpy array (M different initial conditions)
    # return = N x 1 x M numpy array (M different initial conditions)
    def sim_systems(self, A, b, init_cond):
        result = A @ np.moveaxis(init_cond, 2, 0) + b[:,None]
        return np.moveaxis(result, 0, 2)

    # Compute the partial derivatives (gradients) of a multi-dimensional function.
    # dx = Scalar value showing distance between discrete samples
    # y = N-dimentional numpy array of sample points at regular intervals.
    #   -For the 2D case, y is N x N
    # returns = gradient (as torch tensor) at every point in y.
    #   -For the 2D case, return 2 x (N-1) x (N-1).  The first 2 channels are the x and y components
    def compute_derivative(self, dx, y):
        shape = np.array(y.shape) - 1
        deriv = torch.zeros([y.ndim] + list(shape))

        #take deriv along each axis
        y_cropped = y[tuple(y.ndim*[slice(0,-1)])]
        for dim in range(0, y.ndim):
            slice_piece = tuple(dim*[slice(0,-1)] + [slice(1,None)] + (y.ndim-1-dim)*[slice(0,-1)])
            deriv[dim] = torch.tensor((y[slice_piece]-y_cropped)/dx)

        return deriv
