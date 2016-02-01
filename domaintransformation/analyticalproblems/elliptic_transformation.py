# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Simon Camphausen <s_camp02@uni-muenster.de>

from __future__ import absolute_import, division, print_function

from pymor.analyticalproblems.elliptic import EllipticProblem

from domaintransformation.functions.basic import RHSTransformation

from domaintransformation.functions.basic import ProductFunction

from domaintransformation.functions.basic import DomainTransformationFunction
from domaintransformation.parameters.spaces import ProductParameterSpace

from pymor.functions.basic import ConstantFunction






class AffineTransformationProblem(EllipticProblem):
    """Affinely transformed elliptic problem
    """

    def __init__(self, problem, transformation):
        assert isinstance(problem, EllipticProblem)
        assert isinstance(transformation, DomainTransformationFunction)

        self.elliptic_problem = problem
        self.transformation = transformation

        self.domain = problem.domain
        self.rhs = problem.rhs
        self.dirichlet_data = problem.dirichlet_data
        self.neumann_data = problem.neumann_data

        self.name = "DomainTransformation({},{})".format(problem.name, transformation.name)

        elliptic_diffusion_functions = problem.diffusion_functions
        elliptic_diffusion_functionals = problem.diffusion_functionals

        assert elliptic_diffusion_functionals is None

        # TODO Handle diffusion_functionals of the transformation, ProductFunctional is needed.
        # TODO How are functionals and functions separated in the transformation?
        self.diffusion_functions = [ProductFunction(transformation, d_e) for d_e in elliptic_diffusion_functions]
        #self.diffusion_functionals = elliptic_diffusion_functionals
        self.diffusion_functionals = [1.0]

        self.parameter_space = ProductParameterSpace(transformation.parameter_space, problem.parameter_space)

class TransformationProblem(EllipticProblem):
    """Arbitrarily transformed elliptic problem
    """

    def __init__(self, problem, transformation, rhs_transformation):
        assert isinstance(problem, EllipticProblem)
        assert isinstance(transformation, DomainTransformationFunction)
        assert isinstance(rhs_transformation, DomainTransformationFunction)

        self.elliptic_problem = problem
        self.transformation = transformation

        self.domain = problem.domain

        self.dirichlet_data = problem.dirichlet_data
        self.neumann_data = problem.neumann_data

        self.name = "DomainTransformation({},{})".format(problem.name, transformation.name)

        self.rhs = RHSTransformation(problem.rhs, rhs_transformation)

        elliptic_diffusion_functions = problem.diffusion_functions
        elliptic_diffusion_functionals = problem.diffusion_functionals

        assert elliptic_diffusion_functionals is None
        assert len(elliptic_diffusion_functions) == 1

        # TODO Handle diffusion_functionals of the transformation, ProductFunctional is needed.
        # TODO How are functionals and functions separated in the transformation?
        self.diffusion_functions = [ProductFunction(transformation, d_e) for d_e in elliptic_diffusion_functions]
        #self.diffusion_functionals = elliptic_diffusion_functionals
        self.diffusion_functionals = [1.0]

        #self.advection_functions = [ProductFunction(transformation.adv, d_e) for d_e in elliptic_diffusion_functions]
        self.advection_functionals = [1.0]

        self.parameter_space = ProductParameterSpace(transformation.parameter_space, problem.parameter_space)



