from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.linalg import solve_triangular

from pymor.functions.interfaces import FunctionInterface
from pymor.functions.basic import FunctionBase

from pymor.parameters.interfaces import ParameterFunctionalInterface

from domaintransformation.functions.basic import GriddataFunction


class EmpiricalInterpolatedFunction(FunctionBase):

    def __init__(self, function, interpolation_dofs, collateral_basis, xs, triangular, name=None):
        assert isinstance(function, FunctionInterface)
        assert isinstance(collateral_basis, np.ndarray)

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
                    interpolation_matrix.append(collateral_basis[index].T)
                interpolation_matrix = np.array(interpolation_matrix)
            else:
                interpolation_matrix = collateral_basis[:, interpolation_dofs].T
            self.interpolation_matrix = interpolation_matrix
        self.collateral_basis = collateral_basis.copy()

        self.name = name

        # create interpolation operators for discrete basis functions so that these can be evaluated at arbitrary points
        self.operators_linear = [GriddataFunction(self.xs, self.collateral_basis[i, ...], method='linear')
                         for i in range(self.collateral_basis.shape[0])]
        self.operators_nearest = [GriddataFunction(self.xs, self.collateral_basis[i, ...], method='nearest')
                         for i in range(self.collateral_basis.shape[0])]

    def evaluate(self, x, mu=None):
        mu = self.parse_parameter(mu)
        if len(self.interpolation_dofs) == 0:
            return np.zeros_like(self.shape_range).reshape((1,) + self.shape_range).repeat(len(x), axis=0)
        F = self.function(self.xs_interpolation, mu)

        if not self.function.shape_range == tuple():
            F = self.function(self.xs, mu)
            F_ = np.array([F[i, j, k] for i, j, k in self.interpolation_dofs])
            F = F_

        if self.triangular:
            interpolation_coefficients = solve_triangular(self.interpolation_matrix, F,
                                                          lower=True, unit_diagonal=True).T
        else:
            interpolation_coefficients = np.linalg.solve(self.interpolation_matrix, F).T
        assert len(self.collateral_basis) == len(interpolation_coefficients)

        assert self.xs.shape[0] == self.collateral_basis.shape[1]

        x_shape = x.shape

        if x.ndim > 1:
            x_reshape = x.reshape((-1, 2))
        else:
            x_reshape = x

        evaluations_1 = [self.operators_linear[i](x_reshape) for i in range(self.collateral_basis.shape[0])]
        evaluations_2 = [self.operators_nearest[i](x_reshape) for i in range(self.collateral_basis.shape[0])]

        assert len(evaluations_1) == len(evaluations_2)

        for i in range(len(evaluations_1)):
            mask = np.isnan(evaluations_1[i])
            evaluations_1[i][mask] = evaluations_2[i][mask]

        evaluations = evaluations_1

        if len(x_shape) == 3:
            evaluations = [evaluations[i].reshape(x_shape[:-1]) for i in range(len(evaluations))]

        res = np.zeros_like(evaluations[0])

        for i in range(len(interpolation_coefficients)):
            res += interpolation_coefficients[i] * evaluations[i]
        return res

# return interpolation_coefficients[self.index]
class EmpiricalInterpolatedProjectionFunctional(ParameterFunctionalInterface):

    def __init__(self, function, interpolation_dofs, collateral_basis, xs, triangular, index, name=None):
        assert isinstance(function, FunctionInterface)
        assert isinstance(collateral_basis, np.ndarray)
        assert isinstance(index, int)

        assert 0 <= index < len(interpolation_dofs)

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
        self.index = index

        if len(interpolation_dofs) > 0:
            if not function.shape_range == tuple():
                interpolation_matrix = []
                for i in range(len(interpolation_dofs)):
                    index = (slice(None),) + interpolation_dofs[i]
                    interpolation_matrix.append(collateral_basis[index].T)
                interpolation_matrix = np.array(interpolation_matrix)
            else:
                interpolation_matrix = collateral_basis[:, interpolation_dofs].T
            self.interpolation_matrix = interpolation_matrix
        self.collateral_basis = collateral_basis.copy()

        self.name = name

    def evaluate(self, mu=None):
        mu = self.parse_parameter(mu)
        if len(self.interpolation_dofs) == 0:
            return 0.0
        F = self.function(self.xs_interpolation, mu)

        if not self.function.shape_range == tuple():
            F = self.function(self.xs, mu)
            F_ = np.array([F[i, j, k] for i, j, k in self.interpolation_dofs])
            F = F_

        if self.triangular:
            interpolation_coefficients = solve_triangular(self.interpolation_matrix, F,
                                                          lower=True, unit_diagonal=True).T
        else:
            interpolation_coefficients = np.linalg.solve(self.interpolation_matrix, F).T
        assert len(self.collateral_basis) == len(interpolation_coefficients)

        res = interpolation_coefficients[self.index]
        return res


# return evaluations[self.index]
class EmpiricalInterpolatedProjectionFunction(FunctionBase):

    def __init__(self, function, interpolation_dofs, collateral_basis, xs, triangular, index, name=None):
        assert isinstance(function, FunctionInterface)
        assert isinstance(collateral_basis, np.ndarray)
        assert isinstance(index, int)

        assert 0 <= index < len(interpolation_dofs)

        self.dim_domain = function.dim_domain
        self.shape_range = function.shape_range
        self.function = function
        self.interpolation_dofs = interpolation_dofs
        self.xs = xs
        if not function.shape_range == tuple():
            interpolation_xs = tuple(set([dof[0] for dof in interpolation_dofs]))
            interpolation_xs = np.array(interpolation_xs)
            self.xs_interpolation = np.array(xs[interpolation_xs])
        else:
            self.xs_interpolation = xs[interpolation_dofs]
        self.index = index

        self.collateral_basis = collateral_basis.copy()

        self.name = name

        # create interpolation operators for discrete basis functions so that these can be evaluated at arbitrary points
        self.operator_linear = GriddataFunction(self.xs, self.collateral_basis[self.index, ...], method='linear')

        self.operator_nearest = GriddataFunction(self.xs, self.collateral_basis[self.index, ...], method='nearest')


    def evaluate(self, x, mu=None):
        if len(self.interpolation_dofs) == 0:
            return np.zeros_like(self.shape_range).reshape((1,) + self.shape_range).repeat(len(x), axis=0)

        assert self.xs.ndim == 2
        assert self.xs.shape[-1] == 2
        assert self.xs.shape[0] == self.collateral_basis.shape[1]

        x_shape = x.shape
        x_reshape = x.reshape((-1, 2))

        evaluations_1 = self.operator_linear(x_reshape)
        evaluations_2 = self.operator_nearest(x_reshape)

        mask = np.isnan(evaluations_1)
        evaluations_1[mask] = evaluations_2[mask]

        evaluations = evaluations_1

        if len(x_shape) == 3:
            evaluations = evaluations.reshape(x_shape[:-1])

        res = evaluations

        return res
