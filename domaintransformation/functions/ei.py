from __future__ import absolute_import, division, print_function


import numpy as np
from scipy.linalg import solve_triangular

from pymor.functions.interfaces import FunctionInterface
from pymor.functions.basic import LincombFunction
from pymor.vectorarrays.interfaces import VectorArrayInterface


class EmpiricalInterpolatedFunction(FunctionInterface):

    def __init__(self, function, interpolation_dofs, collateral_basis, xs, triangular, name=None):
        assert isinstance(function, FunctionInterface)
        assert isinstance(collateral_basis, np.ndarray)

        #assert collateral_basis.shape[:function.dim_domain] == function.shape_range

        self.dim_domain = function.dim_domain
        self.shape_range = function.shape_range
        self.build_parameter_type(inherits=(function,))



        self.function = function
        self.interpolation_dofs = interpolation_dofs
        self.xs = xs
        if not function.shape_range == tuple():
            interpolation_xs = tuple(set([dof[0] for dof in interpolation_dofs]))
            interpolation_xs = np.array(interpolation_xs)
            self.xs_interpolation = np.array(xs[interpolation_xs])
        else:
            self.xs_interpolation = xs[interpolation_dofs]
        self.triangular = triangular

        if len(interpolation_dofs) > 0:
            if not function.shape_range == tuple():
                interpolation_matrix = []
                for i in range(len(interpolation_dofs)):
                    index = (slice(None),) + interpolation_dofs[i]
                    #interpolation_matrix = collateral_basis[:, interpolation_dofs].T
                    interpolation_matrix.append(collateral_basis[index].T)
                interpolation_matrix = np.array(interpolation_matrix)
            else:
                interpolation_matrix = collateral_basis[:, interpolation_dofs].T
            self.interpolation_matrix = interpolation_matrix
            self.collateral_basis = collateral_basis.copy()

        self.name = name

    def evaluate(self, x, mu=None):
        mu = self.parse_parameter(mu)
        if len(self.interpolation_dofs) == 0:
            return np.zeros_like(self.shape_range).reshape((1,)+ self.shape_range).repeat(len(x), axis=0)
        #F = self.function(x,mu)[self.interpolation_dofs,...]
        F = self.function(self.xs_interpolation, mu)
        if not self.function.shape_range == tuple():
            # FIXME this is not generic
            # FIXME indexing is wrong
            # if xs_interpolation is [0, 10], F has the shape (2,)x...
            # so indexing F in F[10,...] is iwrong
            F_ = np.array([F[i,j,k] for i, j, k in self.interpolation_dofs])
            F = F_

        if self.triangular:
            interpolation_coefficients = solve_triangular(self.interpolation_matrix, F,
                                                              lower=True, unit_diagonal=True).T
        else:
            interpolation_coefficients = np.linalg.solve(self.interpolation_matrix, F).T
        assert len(self.collateral_basis) == len(interpolation_coefficients)

        dim_diff = self.collateral_basis.ndim - 1
        index = (Ellipsis,) + (np.newaxis,)*dim_diff
        res = (self.collateral_basis * interpolation_coefficients[index]).sum(axis=0)
        return res