# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from algorithms.timestepping import imex_euler

from pymor.discretizations.basic import DiscretizationBase
from pymor.operators.constructions import VectorOperator
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.tools.arguments import method_arguments
from pymor.operators.interfaces import OperatorInterface


class InstationaryImexDiscretization(DiscretizationBase):

    def __init__(self, T, nt, initial_data, explicit_operator, implicit_operator, rhs=None, mass=None, num_values=None,
                 products=None, parameter_space=None, estimator=None, visualizer=None, cache_region='disk',
                 name=None):        
        assert isinstance(initial_data, (VectorArrayInterface, OperatorInterface))
        assert not isinstance(initial_data, OperatorInterface) or initial_data.source.dim == 1
        assert isinstance(explicit_operator, OperatorInterface)
        assert isinstance(implicit_operator, OperatorInterface)
        assert rhs is None or isinstance(rhs, OperatorInterface) and rhs.linear
        assert mass is None or isinstance(mass, OperatorInterface) and mass.linear
        if isinstance(initial_data, VectorArrayInterface):
            initial_data = VectorOperator(initial_data, name='initial_data')
        assert explicit_operator.source == explicit_operator.range == implicit_operator.source == implicit_operator.range == initial_data.range
        assert rhs is None or rhs.source == explicit_operator.source and rhs.range.dim == 1
        assert mass is None or mass.source == mass.range == explicit_operator.source

        operators = {'explicit_operator': explicit_operator, 'implicit_operator': implicit_operator, 'mass': mass}
        functionals = {'rhs': rhs}
        vector_operators = {'initial_data': initial_data}
        super(InstationaryImexDiscretization, self).__init__(operators=operators, functionals=functionals,
                                                         vector_operators=vector_operators,
                                                         products=products, estimator=estimator,
                                                         visualizer=visualizer, cache_region=cache_region, name=name)
        
        self.T = T
        self.solution_space = explicit_operator.source
        self.nt = nt
        self.initial_data = initial_data
        self.explicit_operator = explicit_operator
        self.implicit_operator = implicit_operator
        self.rhs = rhs
        self.mass = mass
        self.num_values = num_values
        self.build_parameter_type(inherits=(initial_data, explicit_operator, implicit_operator, rhs, mass), provides={'_t': 0})
        self.parameter_space = parameter_space

    with_arguments = frozenset(method_arguments(__init__)).union({'operators', 'functionals', 'vector_operators'})

    def with_(self, **kwargs):
        assert set(kwargs.keys()) <= self.with_arguments
        assert 'operators' not in kwargs or kwargs['operators'].viewkeys() <= {'explicit_operator', 'implicit_operator', 'mass'}
        assert 'functionals' not in kwargs or kwargs['functionals'].viewkeys() <= {'rhs'}
        assert 'vector_operators' not in kwargs or kwargs['vector_operators'].viewkeys() <= {'initial_data'}
        assert 'operators' not in kwargs or not set(kwargs['operators']).intersection(kwargs.viewkeys())
        assert 'functionals' not in kwargs or not set(kwargs['functionals']).intersection(kwargs.viewkeys())
        assert 'vector_operators' not in kwargs or not set(kwargs['vector_operators']).intersection(kwargs.viewkeys())
        assert 'time_stepper_nt' not in kwargs or 'time_stepper' not in kwargs
        if 'operators' in kwargs:
            kwargs.update(kwargs.pop('operators'))
        if 'functionals' in kwargs:
            kwargs.update(kwargs.pop('functionals'))
        if 'vector_operators' in kwargs:
            kwargs.update(kwargs.pop('vector_operators'))
        if 'time_stepper_nt' in kwargs:
            kwargs['time_stepper'] = self.time_stepper.with_(nt=kwargs.pop('time_stepper_nt'))

        return self._with_via_init(kwargs)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu).copy()

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            if self.linear:
                pt = 'sparsity unknown' if getattr(self.operator, 'sparse', None) is None \
                    else ('sparse' if self.operator.sparse else 'dense')
            else:
                pt = 'nonlinear'
            self.logger.info('Solving {} ({}) for {} ...'.format(self.name, pt, mu))

        mu['_t'] = 0
        U0 = self.initial_data.as_vector(mu)
        return imex_euler(self.explicit_operator, self.implicit_operator, self.rhs, U0, 0, self.T, self.nt, mu)
    