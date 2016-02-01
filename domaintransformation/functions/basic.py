from __future__ import absolute_import, division, print_function


import numpy as np

from pymor.functions.interfaces import FunctionInterface
from pymor.parameters.base import ParameterType

from domaintransformation.parameters.base import ProductParameterType

class RHSTransformation(FunctionInterface):

    def __init__(self, rhs, transformation):

        assert transformation.shape_range == rhs.dim_domain

        self.shape_range = rhs.shape_range
        self.dim_domain = transformation.dim_domain

        self.rhs = rhs
        self.transformation = transformation

        self.build_parameter_type(transformation.parameter_type, local_global=True)

    def evaluate(self, x, mu=None):
        x_ = x.reshape((-1, 2))
        t = self.transformation.evaluate(x_, mu)
        rhs = self.rhs(t, mu)
        j_det = self.transformation.jacobian_determinant(x_, mu)
        res = rhs * j_det
        return res.reshape((x.shape[0],-1))


class ProductFunction(FunctionInterface):
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
            raise NotImplementedError, "One shape_range has to be tuple()"

        self.name = "ProductFunction({},{}".format(function_1.name, function_2.name)

        self.parameter_type = ProductParameterType(function_1.parameter_type, function_2.parameter_type)

    def evaluate(self, x, mu=None):
        x_1 = self.function_1.evaluate(x, mu)
        x_2 = self.function_2.evaluate(x, mu)

        assert x_1.shape[0] == x_2.shape[0] # outermost shape must match

        len_1 = len(x_1.shape)
        len_2 = len(x_2.shape)

        if len_1==1 and len_2==1:
            return x_1*x_2
        elif len_1==1:
            assert len_2 > 1
            dim_diff = len_2 - len_1
            index = (Ellipsis,) + (np.newaxis,) * dim_diff
            return x_1[index]*x_2
        elif len_2==1:
            assert len_1 > 1
            dim_diff = len_1 - len_2
            index = (Ellipsis,) + (np.newaxis,) * dim_diff
            # x_2_view = x_2[index]
            return x_1*x_2[index]
        else:
            raise NotImplementedError


class DomainTransformationFunction(FunctionInterface):
    """A |Function| representing an arbitrary transformation of  the domain.
    """

    def __init__(self, parameter_type, name=None):
        self.build_parameter_type(parameter_type, local_global=True)
        self.name = name

    def evaluate(self, x, mu=None):
        return self.diffusion(x, mu)

    def diffusion(self, x, mu=None):
        raise NotImplementedError

    def advection(self, x, mu=None):
        raise NotImplementedError

    def transform(self, x, mu=None):
        raise NotImplementedError

    def bounding_box(self, domain, mu):
        raise NotImplementedError
