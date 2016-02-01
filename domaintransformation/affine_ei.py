from __future__ import absolute_import, division, print_function

import sys
import math as m
import time
from functools import partial

import numpy as np

from domaintransformation.analyticalproblems.elliptic_transformation import AffineTransformationProblem
from pymor.analyticalproblems.elliptic import EllipticProblem
from domaintransformation.functions.basic import DomainTransformationFunction
from pymor.parameters.interfaces import ParameterSpaceInterface
from pymor.parameters.spaces import CubicParameterSpace
from pymor.grids.tria import TriaGrid
from domaintransformation.grids.domaintransformation import DomainTransformationTriaGrid
from domaintransformation.discretizers.elliptic import discretize_elliptic_cg
from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.parameters.base import Parameter
from domaintransformation.algorithms.ei import interpolate_function, interpolate_operators

from domaintransformation.analyticalproblems.affine_transformation import AffineProblem


def demo():
    # Konvergenz zu analytischer Loesung
    cache_region = None
    num_intervals = (5, 5) # Matrizen vergleichen
    domain = np.array([[0,0],[1,1]])
    ei_snapshots = 3
    ei_size = 2
    projection = 'ei'

    separate_colorbars = False

    assert projection in ('orthogonal', 'ei')

    plot_solutions = False
    plot_ei_solutions = True


    print('Setup Problem ...')
    trafo_problem = AffineProblem(min=0.5, max=1.5, name="AffineTrafo")

    assert len(trafo_problem.diffusion_functions) == 1
    transformation = trafo_problem.diffusion_functions[0]

    print('Setup grids ...')
    elliptic_grid = TriaGrid(num_intervals)
    trafo_grid = DomainTransformationTriaGrid(elliptic_grid, transformation)

    print('Discretize ...')
    discretization, discretization_data = discretize_elliptic_cg(trafo_problem, grid=trafo_grid,
                                                     boundary_info=AllDirichletBoundaryInfo(trafo_grid))

    if cache_region is None:
        discretization.disable_caching()

    print('The parameter type is {}'.format(discretization.parameter_type))

    if plot_solutions:
        print('Showing some solutions')
        mus = tuple()
        Us = tuple()
        legend = tuple()
        for mu in discretization.parameter_space.sample_randomly(2):
            mus = mus + (mu,)
            print('Solving for transformation = {} ... '.format(mu['transformation']))
            sys.stdout.flush()
            Us = Us + (discretization.solve(mu),)
            legend = legend + ('Transformation: {}'.format(mu['transformation']),)
        discretization.visualize(Us, mu=mus, legend=legend, title='Detailed Solutions', block=True)

    print("Interpolating Diffusion Operator")

    ei_snapshots = 4**ei_snapshots
    function = discretization.operator.diffusion_function
    samples = discretization.parameter_space.sample_randomly(ei_snapshots)
    g = discretization_data['grid']
    x = g.centers(0)

    #interpolate_operators(discretization, ['operator'], samples)
    function_ei = interpolate_function(function, samples, x, max_interpolation_dofs=10)

    z=0




    mu = {'transformation': np.array([[2,0],[0,2]])}
    mu = Parameter(mu)

    U_trafo = discretization.solve(mu=mu)
    #U_ei = ei_discretization.solve(mu=mu)

    trafo_grid.visualize(U_trafo, mu=mu)
    #trafo_grid.visualize(U_ei, mu=mu)

    #ei_discretization.visualize(U_ei, mu=mu)

    #discretization.visualize((U_trafo, U_ei), mu=(mu,mu), legend=("FEM", "EI"), separate_colorbars=separate_colorbars)


demo()