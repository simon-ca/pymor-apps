# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from itertools import izip, product
import numpy as np

from pymor.parameters.base import Parameter, ParameterType
from pymor.parameters.interfaces import ParameterSpaceInterface
from pymor.tools.random import new_random_state
from pymor.parameters.spaces import CubicParameterSpace
from domaintransformation.parameters.base import ProductParameterType


class ProductParameterSpace(CubicParameterSpace):
    """Product of two CubicParameterSpaces

    Parameters
    ----------
    parameter_space_1
        |ParameterSpace|
    parameter_space_2
        |ParameterSpace|
    """

    def __init__(self, parameter_space_1, parameter_space_2):

        assert parameter_space_1 is None or isinstance(parameter_space_1, CubicParameterSpace)
        assert parameter_space_2 is None or isinstance(parameter_space_2, CubicParameterSpace)
        assert parameter_space_1 is not None or parameter_space_2 is not None, \
            "At least one parameter space must not be None"

        self.parameter_space_1 = parameter_space_1
        self.parameter_space_2 = parameter_space_2

        if parameter_space_1 is None:
            ranges = parameter_space_2.ranges
            parameter_type = parameter_space_2.parameter_type
        elif parameter_space_2 is None:
            ranges = parameter_space_1.ranges
            parameter_type = parameter_space_1.parameter_type
        else:
            parameter_type = ProductParameterType(parameter_space_1.parameter_type, parameter_space_2.parameter_type)
            ranges = parameter_space_1.ranges.update(parameter_space_2.ranges)
            
        super(ProductParameterSpace, self).__init__(parameter_type, ranges=ranges)

    #def parse_parameter(self, mu):
    #    raise NotImplementedError

        #return Parameter.from_parameter_type(mu, self.parameter_type)

    #def contains(self, mu):
    #    mu_1 = {k: v for k, v in mu if k in self.parameter_names_1}
    #    mu_2 = {k: v for k, v in mu if k in self.parameter_names_2}
    #    return self.parameter_space_1.contains(mu_1) and self.parameter_space_2.contains(mu_2)

    #def sample_uniformly(self, counts):
    #    """Iterator sampling uniformly |Parameters| from the space."""
    #    if isinstance(counts, dict):
    #        pass
    #    elif isinstance(counts, (tuple, list, np.ndarray)):
    #        counts = {k: c for k, c in izip(self.parameter_type_1, counts)}
    #        counts.update({k: c for k, c in izip(self.parameter_type_2, counts)})
    #    else:
    #        counts = {k: counts for k in self.parameter_type_1}
    #        counts.update({k: counts for k in self.parameter_type_2})
    #    linspaces = tuple(np.linspace(self.ranges[k][0], self.ranges[k][1], num=counts[k]) for k in self.parameter_type)
    #    iters = tuple(product(ls, repeat=max(1, np.zeros(sps).size))
    #                  for ls, sps in izip(linspaces, self.parameter_type.values()))
    #    for i in product(*iters):
    #        yield Parameter(((k, np.array(v).reshape(shp))
    #                         for k, v, shp in izip(self.parameter_type, i, self.parameter_type.values())))

    #def sample_randomly(self, count=None, random_state=None, seed=None):
    #    raise NotImplementedError

    #    """Iterator sampling random |Parameters| from the space."""
    #    assert not random_state or seed is None
    #    c = 0
    #    ranges = self.ranges
    #    random_state = random_state or new_random_state(seed)
    #    while count is None or c < count:
    #        yield Parameter(((k, random_state.uniform(ranges[k][0], ranges[k][1], shp))
    #                         for k, shp in self.parameter_type.iteritems()))
    #        c += 1

    def __str__(self):
        rows = [(k, str(v), str(self.ranges[k])) for k, v in self.parameter_type.iteritems()]
        column_widths = [max(map(len, c)) for c in zip(*rows)]
        return ('CubicParameterSpace\n' +
                '\n'.join(('key: {:' + str(column_widths[0] + 2)
                           + '} shape: {:' + str(column_widths[1] + 2)
                           + '} range: {}').format(c1 + ',', c2 + ',', c3) for (c1, c2, c3) in rows))

    def __repr__(self):
        return 'ProductParameterSpace({}, {})'.format(repr(self.parameter_space_1), repr(self.parameter_space_2))
