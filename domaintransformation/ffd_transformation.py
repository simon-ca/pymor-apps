from __future__ import absolute_import, division, print_function

import sys

import numpy as np

from domaintransformation.analyticalproblems.elliptic_transformation import AffineTransformationProblem
from pymor.analyticalproblems.elliptic import EllipticProblem
from domaintransformation.functions.basic import DomainTransformationFunction
from pymor.parameters.spaces import CubicParameterSpace
from pymor.grids.tria import TriaGrid
from domaintransformation.grids.domaintransformation import DomainTransformationTriaGrid
from domaintransformation.discretizers.elliptic import discretize_elliptic_cg
from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.parameters.base import Parameter
from pymor.domaindescriptions.basic import RectDomain
from pymor.algorithms.ei import interpolate_operators

def parameter_mask(num_control_points, active=[]):
    assert isinstance(num_control_points, tuple)
    assert len(num_control_points) == 2

    K, L = num_control_points

    assert isinstance(active, list)
    assert len(active) <= K*L*2

    assert all(isinstance(a, tuple) for a in active)
    assert all(len(a) == 3 for a in active)
    assert all(isinstance(x, int) and isinstance(y, int) and isinstance(z, int) for x, y, z in active)
    assert all(x >= 0 and x < K and y >= 0 and y < L and z in [0, 1] for x, y, z in active)

    mask = np.zeros((K, L, 2), dtype=bool)
    for x,y,z in active:
        mask[x,y,z] = True

    return mask

def demo():
    cache_region = None
    num_intervals = (50, 50)
    domain = np.array([[0,0],[1,1]])
    shift_min = -0.5
    shift_max = 0.5
    ei_snapshaots = 3
    ei_size = 3
    separate_colorbars = True

    plot_solutions = False
    plot_ei_solutions = True


    num_control_points = (5, 5)
    active = [(4, 1, 0), (0, 4, 0)]
    mask = parameter_mask(num_control_points, active)

    print('Setup Problems ...')
    elliptic_problem = EllipticProblem(name="elliptic")
    transformation = FFDTransformation(RectDomain(domain), mask, "ffd", shift_min, shift_max)
    trafo_problem = AffineTransformationProblem(elliptic_problem, transformation)

    print('Setup grids ...')
    elliptic_grid = TriaGrid(num_intervals)
    trafo_grid = DomainTransformationTriaGrid(elliptic_grid, transformation)

    print('Discretize ...')
    discretization, _ = discretize_elliptic_cg(trafo_problem, grid=trafo_grid, boundary_info=AllDirichletBoundaryInfo(trafo_grid))

    if cache_region is None:
        discretization.disable_caching()

    print('The parameter type is {}'.format(discretization.parameter_type))

    if plot_solutions:
        print('Showing some solutions')
        mus = tuple()
        Us = tuple()
        legend = tuple()
        for mu in discretization.parameter_space.sample_uniformly(2):
            mus = mus + (mu,)
            print('Solving for transformation = {} ... '.format(mu['ffd']))
            sys.stdout.flush()
            Us = Us + (discretization.solve(mu),)
            legend = legend + ('Transformation: {}'.format(mu['ffd']),)
        discretization.visualize(Us, mu=mus, legend=legend, title='Detailed Solutions', block=True)

    print("Interpolating Diffusion Operator")

    ei_discretization, ei_data = interpolate_operators(discretization, ['operator'],
                                                       discretization.parameter_space.sample_uniformly(ei_snapshaots),
                                                       error_norm=discretization.l2_norm,
                                                       target_error=1e-15,
                                                       max_interpolation_dofs=ei_size,
                                                       projection='orthogonal',
                                                       product=discretization.l2_product)

    if plot_ei_solutions:
        print("Showing some interpolated solutions")
        mus = tuple()
        Us = tuple()
        legend = tuple()
        samples = discretization.parameter_space.sample_randomly(2)
        for mu in samples:
            mus = mus + (mu, mu)
            legend = legend + ("FEM: {}".format(mu['ffd']), "EI: {}".format(mu['ffd']))
            print("Solving for parameter = {} ...".format(mu['ffd']))
            sys.stdout.flush()
            Us = Us + (discretization.solve(mu), ei_discretization.solve(mu)) # ei direkte invertierung
        discretization.visualize(Us, mu=mus, legend=legend, title="Comparison", block=True, separate_colorbars=separate_colorbars)

    mu = {'ffd': np.array([0.15, 0.15])}
    mu = Parameter(mu)

    U_trafo = discretization.solve(mu)
    U_ei = ei_discretization.solve(mu)

    ei_discretization.visualize((U_trafo,U_ei), mu=(mu,mu), legend=("FEM", "EI"), title="Comparison", separate_colorbars=separate_colorbars)


class FFDTransformation(DomainTransformationFunction):
    dim_domain = 2
    shape_range = (2, 2)

    def __init__(self, domain, control_point_mask, parameter_name, min, max, ranges=None):
        assert isinstance(domain, RectDomain)

        assert isinstance(control_point_mask, np.ndarray)
        assert control_point_mask.dtype == np.bool
        assert len(control_point_mask.shape) == 3
        assert control_point_mask.shape[2] == 2

        assert isinstance(parameter_name, str)

        assert min is not None and max is not None and ranges is None

        self.domain = domain
        self.control_point_mask = control_point_mask
        self.parameter_name = parameter_name

        self.mask_index = control_point_mask == True

        self.num_control_points = (control_point_mask.shape[0], control_point_mask.shape[1])

        num_x_points = control_point_mask.shape[0]
        num_y_points = control_point_mask.shape[1]

        parameter_shape = (np.count_nonzero(control_point_mask),)

        x = np.linspace(0, 1, num_x_points)
        y = np.linspace(0, 1, num_y_points)

        self.P_0 = np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))]).reshape((len(x), len(y), 2))

        parameter_type = {parameter_name: parameter_shape}

        super(FFDTransformation, self).__init__(parameter_type)

        self.parameter_space = CubicParameterSpace(parameter_type, min, max, ranges)



    def evaluate(self, x, mu=None):
        assert mu is None or isinstance(mu, Parameter)
        if mu is None:
            return x
        else:
            mu = self._assemble_parameter(mu)
        J = self.jacobian(x, mu)
        det = np.linalg.det(J)
        assert np.count_nonzero(det) == det.size, "Singular value"
        inv = np.linalg.inv(J)
        res_ = np.einsum("eij,ekj,e->eik", inv, inv, np.abs(det)) # second inv is transposed
        return res_

    def transform(self, x, mu=None):
        assert mu is None or isinstance(mu, Parameter)
        if mu is None:
            return x
        if isinstance(mu, Parameter):
            mu = self._assemble_parameter(mu)
        P_mu = self.P_0 + mu

        x_ = self._bernstein2d_vec(self._psi(x))
        x_mu_sum = (x_[..., np.newaxis] * P_mu[np.newaxis, ...]).sum(axis=(1,2))

        return self._psi_inverse(x_mu_sum)

    def jacobian(self, x, mu=None):
        if mu is None:
            return np.eye(2).reshape((1, 2, 2)).repeat(x.shape[0], axis=0)
        x_ = self._bernstein2d_derivative_vec(self._psi(x))
        x_mu = x_[..., np.newaxis] * mu[np.newaxis, ...]
        x_sum = x_mu.sum(axis=(2, 3)) + np.eye(2).reshape((1,2,2))

        psi = self._psi_jacobian()
        psi_inv = self._psi_inverse_jacobian()
        return np.einsum("ij,ejk,kl->eil", psi_inv, x_sum, psi)

    def jacobian_inverse(self, x, mu=None):
        pass

    def bounding_box(self, domain, mu):
        assert isinstance(mu, Parameter) or isinstance(mu, tuple) and all(isinstance(m, Parameter) for m in mu)
        assert isinstance(domain, np.ndarray)

        ll = domain[0,:]
        lr = np.array([domain[1,0], domain[0,1]])
        ul = np.array([domain[0,0], domain[1,1]])
        ur = domain[1,:]

        box = np.array([ll, lr, ul, ur])

        mu = (mu,) if isinstance(mu, Parameter) else mu

        box_transformed = []
        for m in mu:
            l = self._assemble_parameter(m)
            m = self.P_0 + l
            box_transformed.append(m)
        box_transformed = np.array(box_transformed)

        max = box_transformed.max(axis=(0,1,2))
        min = box_transformed.min(axis=(0,1,2))

        return np.array([min, max])

    def _assemble_parameter(self, mu):
        assert isinstance(mu, Parameter)
        mu = mu[self.parameter_name]
        parameter = np.zeros_like(self.control_point_mask, dtype=np.float)
        parameter[self.mask_index] = np.array(mu)
        return parameter

    def _psi(self, X):
        return (X - self.domain.lower_left) / np.array([self.domain.width, self.domain.height])

    def _psi_inverse(self, X):
        return X * np.array([self.domain.width, self.domain.height]) + self.domain.lower_left

    def _psi_jacobian(self):
        return np.diag([1. / self.domain.width, 1. / self.domain.height])

    def _psi_inverse_jacobian(self):
        return np.diag([self.domain.width, self.domain.height])

    def _bernstein1d_vec(self, X, direction, shift_up=0, shift_down=0):
        from scipy.misc import comb
        assert len(X.shape) == 1
        num = self.num_control_points[direction]
        k = np.arange(num) + shift_down
        K = np.ones(num) * (num - 1 + shift_up)

        combs = comb(K, k, exact=False)

        X_ = (1. - X[..., np.newaxis])**(K-k) * X[..., np.newaxis]**k
        return combs[np.newaxis, ...] * X_

    def _bernstein1d_derivative_vec(self, X, index):
        assert len(X.shape) == 1

        b_1 = self._bernstein1d_vec(X, index, -1, -1)
        b_2 = self._bernstein1d_vec(X, index, -1, 0)

        L = self.num_control_points[index]

        return L * (b_1 - b_2)

    def _bernstein2d_vec(self, X):
        assert len(X.shape) == 2

        x = X[..., 0]
        y = X[..., 1]

        b_x = self._bernstein1d_vec(x, 0)
        b_y = self._bernstein1d_vec(y, 1)

        return b_x[:, :, np.newaxis] * b_y[:, np.newaxis, :]

    def _bernstein2d_derivative_vec(self, X):
        assert len(X.shape) == 2

        x = X[..., 0]
        y = X[..., 1]

        b_x = self._bernstein1d_vec(x, 0)
        b_y = self._bernstein1d_vec(y, 1)
        b_x_d = self._bernstein1d_derivative_vec(x, 0)
        b_y_d = self._bernstein1d_derivative_vec(y, 1)

        d_x = b_x_d[:, :, np.newaxis] * b_y[:, np.newaxis, :]
        d_y = b_x[:, :, np.newaxis] * b_y_d[:, np.newaxis, :]

        return np.hstack((d_x[:, np.newaxis, :, :], d_y[:, np.newaxis, :, :]))

demo()