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
from pymor.algorithms.ei import interpolate_operators


def demo():
    # Konvergenz zu analytischer Loesung
    cache_region = None
    num_intervals = (2, 2)
    domain = np.array([[0,0],[1,1]])
    ei_snapshots = 3
    ei_size = 10
    projection = 'ei'

    separate_colorbars = False

    assert projection in ('orthogonal', 'ei')

    plot_solutions = False
    plot_ei_solutions = True


    print('Setup Problems ...')
    elliptic_problem = EllipticProblem(name="elliptic")
    transformation_type = {'transformation': (2,2)}
    transformation = AffineTransformation(transformation_type, 0.5, 1.5)
    trafo_problem = AffineTransformationProblem(elliptic_problem, transformation)

    print('Setup grids ...')
    elliptic_grid = TriaGrid(num_intervals)
    trafo_grid = DomainTransformationTriaGrid(elliptic_grid, transformation)

    print('Discretize ...')
    discretization, _ = discretize_elliptic_cg(trafo_problem, grid=trafo_grid,
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

    ei_discretization, ei_data = interpolate_operators(discretization, ['operator'],
                                                       discretization.parameter_space.sample_randomly(ei_snapshots),
                                                       error_norm=discretization.l2_norm,
                                                       target_error=1e-10,
                                                       max_interpolation_dofs=ei_size,
                                                       projection=projection,
                                                       product=discretization.l2_product)

    mu = {'transformation': np.array([[2,0],[0,2]])}
    mu = Parameter(mu)

    U_trafo = discretization.solve(mu=mu)
    U_ei = ei_discretization.solve(mu)

    #trafo_grid.visualize(U_trafo, mu=mu)
    #trafo_grid.visualize(U_ei, mu=mu)

    #ei_discretization.visualize(U_ei, mu=mu)

    z = 0
    discretization.visualize((U_trafo, U_ei), mu=(mu,mu), legend=("FEM", "EI"), separate_colorbars=separate_colorbars)


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






demo()