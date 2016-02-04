from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.linalg import solve_triangular

from pymor.functions.interfaces import FunctionInterface
from pymor.functions.basic import FunctionBase

from pymor.parameters.base import Parameter
from pymor.functions.basic import LincombFunction
from pymor.vectorarrays.interfaces import VectorArrayInterface


class FixedParameterFunction(FunctionBase):

    def __init__(self, function, mu):
        assert isinstance(function, FunctionBase)
        assert isinstance(mu, Parameter)
        self.dim_domain = function.dim_domain
        self.shape_range = function.shape_range
        self.function = function
        self.mu = mu

    def evaluate(self, x, mu=None):
        #assert mu is None
        return self.function(x, self.mu)


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
            F = self.function(self.xs, mu)
            # FIXME this is not generic
            # FIXME indexing is wrong
            # if xs_interpolation is [0, 10], F has the shape (2,)x...
            # so indexing F in F[10,...] is iwrong
            #z = np.unique(np.array([i for i,_,_ in self.interpolation_dofs]))
            #y = [(np.where(z==i)[0][0],j,k) for i,j,k in self.interpolation_dofs]
            F_ = np.array([F[i,j,k] for i,j,k in self.interpolation_dofs])
            #F_ = np.array([F[i,j,k] for i, j, k in y])

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
        #res = res[:,np.newaxis,:].repeat()
        if x.ndim == 3:
            res = np.tile(res[:,np.newaxis], (1,x.shape[1]))
        return res


class EIFunction(FunctionBase):

    def __init__(self, function, basis_functions, interpolation_points):
        assert isinstance(function, FunctionInterface)
        assert isinstance(basis_functions, list)
        assert all([isinstance(basis_function, FunctionInterface) for basis_function in basis_functions])
        assert isinstance(interpolation_points, list)
        assert all([isinstance(interpolation_point, float) for interpolation_point in interpolation_points])

        assert len(basis_functions) == len(interpolation_points)


        self.dim_domain = function.dim_domain
        self.shape_range = function.shape_range

        self.function = function
        self.basis_functions = basis_functions
        self.interpolation_points = interpolation_points

        l = len(basis_functions)
        if l > 1:
            z = 0
        interpolation_matrix = np.zeros(shape=(l,l), dtype=np.float)
        for i in range(l):
            for j in range(l):
                interpolation_matrix[i,j] = basis_functions[i](interpolation_points[j])
        assert interpolation_matrix.shape == (len(self.interpolation_points),)*2
        self.interpolation_matrix = interpolation_matrix

    def evaluate(self, x, mu=None):
        fs = [self.function(self.interpolation_points[i], mu) for i in range(len(self.interpolation_points))]
        fs = np.array(fs)

        if len(fs) > 1:
            z = 0

        phis = np.linalg.inv(self.interpolation_matrix)*fs
        phis = phis[0,:]

        assert len(self.interpolation_points) == len(self.basis_functions)
        #assert self.interpolation_matrix.shape == (len(self.basis_functions),)*2
        #assert fs.shape*2 == self.interpolation_matrix.shape


        assert phis.shape == fs.shape

        l = [phis[i]*self.basis_functions[i](x) for i in range(len(self.interpolation_points))]
        s = sum(l)
        return s