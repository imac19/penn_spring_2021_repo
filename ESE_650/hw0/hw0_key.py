import torch
import numpy as np

class HW0Solution:
    def __init__(self):
        pass

    def compute_derivative(self, dx, y):
        shape = np.array(y.shape) - 1
        deriv = torch.zeros([y.ndim] + list(shape))

        #take deriv along each axis
        y_cropped = y[tuple(y.ndim*[slice(0,-1)])]
        for dim in range(0, y.ndim):
            slice_piece = tuple(dim*[slice(0,-1)] + [slice(1,None)] + (y.ndim-1-dim)*[slice(0,-1)])
            deriv[dim] = torch.tensor((y[slice_piece]-y_cropped)/dx)

        return deriv

    def sim_systems(self, A, b, init_cond):
        result = A @ np.moveaxis(init_cond,2,0) + b[:,None]
        return np.moveaxis(result,0,2)
