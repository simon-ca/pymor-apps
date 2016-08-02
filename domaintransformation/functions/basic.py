from __future__ import absolute_import, division, print_function

from pymor.functions.basic import FunctionBase, FunctionInterface

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from scipy.interpolate.interpnd import LinearNDInterpolator
from scipy.interpolate import interp1d

from domaintransformation.parameters.base import ProductParameterType


class ProjectionFunction(FunctionBase):
    """
    Returns the evaluations of one compinent of a function
    """
    def __init__(self, function, index):
        # todo lift this restriction
        assert function.shape_range == (2, 2)

        assert isinstance(index, tuple)
        assert (0, 0) <= index <= (1, 1)

        self.function = function
        self.index = index

        self.dim_domain = function.dim_domain
        self.shape_range = tuple()
        self.build_parameter_type(inherits=(function,))

    def evaluate(self, x, mu=None):
        assert x.ndim == 2
        f = self.function.evaluate(x, mu)
        assert f.ndim == 3
        i, j = self.index
        return f[:, i, j]


class WideningFunction(FunctionBase):
    """
    Gets a Function of shape range tuple(), a shape range and an index.
    Evaluation at index is the evaluation of the Function otherwise it is zero.
    """

    def __init__(self, function, shape_range, index):
        assert isinstance(function, FunctionBase)
        assert function.shape_range == tuple()
        assert isinstance(shape_range, tuple)
        assert all(0 < r for r in shape_range)
        assert isinstance(index, tuple)
        assert all(0 <= r for r in index)
        assert len(shape_range) == len(index)
        assert index <= shape_range

        # todo lift this restriction
        assert shape_range == (2, 2)

        self.function = function
        self.index = index

        self.shape_range = shape_range
        self.dim_domain = function.dim_domain
        self.build_parameter_type(inherits=(function,))

    def evaluate(self, x, mu=None):
        assert x.ndim in (1, 2)
        x_shape = x.shape[0]
        res = np.zeros(shape=(x_shape, 2, 2))

        i, j = self.index
        res[:, i, j] = self.function.evaluate(x, mu)

        return res


class MergeFunction(FunctionBase):
    def __init__(self, func_dict):
        assert isinstance(func_dict, dict)
        assert all(isinstance(key, tuple) for key in func_dict.keys())
        assert all(isinstance(func, FunctionInterface) for func in func_dict.values())
        assert all(func.dim_domain == func_dict.values()[0].dim_domain for func in func_dict.values())
        assert all(func.shape_range == tuple() for func in func_dict.values())
        assert all(len(key) == len(func_dict.keys()[0]) for key in func_dict.keys())
        self.func_dict = func_dict

        # todo lift this restriction
        assert len(func_dict) == 4
        assert (0, 0) in func_dict.keys()
        assert (0, 1) in func_dict.keys()
        assert (1, 0) in func_dict.keys()
        assert (1, 1) in func_dict.keys()

        self.dim_domain = func_dict.values()[0].dim_domain
        self.shape_range = (2, 2)
        self.build_parameter_type(inherits=(func_dict.values()[0],))

    def evaluate(self, x, mu=None):
        assert x.ndim in (1, 2)
        x_shape = x.shape[0]
        res = np.empty(shape=(x_shape, 2, 2))
        res[:, 0, 0] = self.func_dict[(0, 0)].evaluate(x, mu)
        res[:, 0, 1] = self.func_dict[(0, 1)].evaluate(x, mu)
        res[:, 1, 0] = self.func_dict[(1, 0)].evaluate(x, mu)
        res[:, 1, 1] = self.func_dict[(1, 1)].evaluate(x, mu)
        return res


class ProductFunction(FunctionBase):
    """A |Function| representing the product of two |Functions|.
    Parameters
    ----------
    function_1
        |Function|
    function_2
        |Function|
    Attributes
    ----------
    function_1
    function_2
    """

    def __init__(self, function_1, function_2, name=None):

        assert function_1.dim_domain == function_2.dim_domain

        self.function_1 = function_1
        self.function_2 = function_2
        self.dim_domain = function_1.dim_domain
        if function_1.shape_range == tuple():
            self.shape_range = function_2.shape_range
        elif function_2.shape_range == tuple():
            self.shape_range = function_1.shape_range
        else:
            raise NotImplementedError("One shape_range has to be tuple()")

        self.name = "ProductFunction({},{})".format(function_1.name, function_2.name)

        self.parameter_type = ProductParameterType(function_1.parameter_type, function_2.parameter_type)

    def evaluate(self, x, mu=None):
        x_1 = self.function_1.evaluate(x, mu)
        x_2 = self.function_2.evaluate(x, mu)

        assert x_1.shape[0] == x_2.shape[0]  # outermost shape must match

        len_1 = len(x_1.shape)
        len_2 = len(x_2.shape)

        if len_1 == 1 and len_2 == 1:
            return x_1 * x_2
        elif len_1 == 1:
            assert len_2 > 1
            dim_diff = len_2 - len_1
            index = (Ellipsis,) + (np.newaxis,) * dim_diff
            return x_1[index] * x_2
        elif len_2 == 1:
            assert len_1 > 1
            dim_diff = len_1 - len_2
            index = (Ellipsis,) + (np.newaxis,) * dim_diff
            # x_2_view = x_2[index]
            return x_1 * x_2[index]
        elif x_1.shape == x_2.shape:
            # fixme this is a hack
            return x_1 * x_2
            raise NotImplementedError


class GriddataFunction(FunctionBase):
    """A |Function| which uses scipy.interpolate.griddata to interpolate a function with
        given evaluations at some points
    """

    def __init__(self, points, values, method):
        assert isinstance(points, np.ndarray)
        assert points.ndim in [1, 2]
        assert isinstance(values, np.ndarray)
        # assert evaluations.ndim == 1
        assert points.shape[0] == values.shape[0]
        assert isinstance(method, str)
        assert method in ['nearest', 'linear']

        self.points = points
        self.values = values
        self.method = method

        if points.ndim == 2:
            self.mode = '2d'
            ndim = points.shape[-1]
            assert ndim == 2
            if method == 'nearest':
                self.operator = NearestNDInterpolator(points, values)
            elif method == 'linear':
                fill_value = np.nan
                self.operator = LinearNDInterpolator(points, values, fill_value=fill_value)
        else:
            self.mode = '1d'
            if method == 'nearest':
                self.operator = interp1d(points, values, kind='nearest')
            elif method == 'linear':
                self.operator = interp1d(points, values, kind='linear')

    def evaluate(self, x, mu=None):
        return self.operator(x)