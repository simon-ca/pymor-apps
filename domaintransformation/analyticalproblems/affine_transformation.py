# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

from __future__ import absolute_import, division, print_function

from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction, GenericFunction

from pymor.parameters.spaces import CubicParameterSpace
from pymor.parameters.base import Parameter

from domaintransformation.functions.basic import DomainTransformationFunction

import numpy as np


class AffineProblem(EllipticProblem):
    """Affinely transformed elliptic problem.

    Parameters
    ----------
    domain
        A |DomainDescription| of the domain the problem is posed on.
    rhs
        The |Function| f(x, μ). `rhs.dim_domain` has to agree with the
        dimension of `domain`, whereas `rhs.shape_range` has to be `tuple()`.
    diffusion_functions
        List containing the |Functions| d_k(x), each having `shape_range`
        of either `tuple()` or `(dim domain, dim domain)`.
    diffusion_functionals
        List containing the |ParameterFunctionals| θ_k(μ). If
        `len(diffusion_functions) == 1`, `diffusion_functionals` is allowed
        to be `None`, in which case no parameter dependence is assumed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values in global coordinates.
    neumann_data
        |Function| providing the Neumann boundary values in global coordinates.
    min
        Minimum for transformation.
    max
        Maximum for transformation.
    name
        Name of the problem.

    Attributes
    ----------
    domain
    rhs
    diffusion_functions
    diffusion_functionals
    dirichlet_data
    """

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 diffusion_functions=(ConstantFunction(dim_domain=2),),
                 diffusion_functionals=None,
                 dirichlet_data=None, neumann_data=None,
                 min=0.5, max=1.5, name=None):

        parameter_type = {'transformation': (2,2)}

        super(AffineProblem, self).__init__(domain, rhs, diffusion_functions, diffusion_functionals, dirichlet_data,
                                            neumann_data, name)
        self.min = min
        self.max = max

        self.parameter_space = CubicParameterSpace(parameter_type, minimum=min, maximum=max)

        self.diffusion_functions = [AffineTransformation(parameter_type, min, max)]
        self.diffusion_functionals = None


class AffineTransformation(DomainTransformationFunction):
    dim_domain = 2
    shape_range = (2, 2)

    def __init__(self, parameter_type, min=None, max=None, ranges=None):
        assert isinstance(parameter_type, dict)
        assert len(parameter_type) == 1

        self.transformation_name = parameter_type.keys()[0]
        assert isinstance(self.transformation_name, str)

        super(AffineTransformation, self).__init__(parameter_type)

        self.parameter_space = CubicParameterSpace(parameter_type, min, max, ranges)

    def evaluate(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        det = np.linalg.det(mu)
        assert not det == 0

        A_inv = np.linalg.inv(mu)

        res = A_inv.dot(A_inv.T)*np.abs(det)

        return res.reshape((1, 2, 2)).repeat(x.shape[0], axis=0)

    def transform(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        assert len(x.shape) == 2
        return np.einsum("ij,ej->ei", mu, x)

    def jacobian(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        return mu

    def jacobian_inverse(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        return np.linalg.inv(self.jacobian(x, mu))

    def bounding_box(self, domain, mu):
        assert isinstance(mu, Parameter) or isinstance(mu, tuple) and all(isinstance(m, Parameter) for m in mu)
        assert isinstance(domain, np.ndarray)

        ll = domain[0,:]
        lr = np.array([domain[1,0], domain[0,1]])
        ul = np.array([domain[0,0], domain[1,1]])
        ur = domain[1,:]

        box = np.array([ll, lr, ul, ur])

        mu = (mu,) if isinstance(mu, Parameter) else mu

        box_transformed = np.array([self.transform(box, m) for m in mu])

        max_x = box_transformed[:,:,0].max(axis=(0,1))
        max_y = box_transformed[:,:,1].max(axis=(0,1))
        min_x = box_transformed[:,:,0].min(axis=(0,1))
        min_y = box_transformed[:,:,1].min(axis=(0,1))

        max = box_transformed.max(axis=(0,1))
        min = box_transformed.min(axis=(0,1))

        box_1 = np.array([[min_x, min_y],[max_x, max_y]])
        box_2 = np.array([min, max])

        return box_1
