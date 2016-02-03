from __future__ import absolute_import, division, print_function

from pymor.functions.basic import FunctionBase

import numpy as np

from domaintransformation.parameters.base import ProductParameterType

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
        elif x_1.shape == x_2.shape:
            #fixme this is a hack
            return x_1*x_2
            raise NotImplementedError