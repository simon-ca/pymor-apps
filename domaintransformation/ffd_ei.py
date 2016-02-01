import numpy as np
import matplotlib.pyplot as plt

from pymor.grids.tria import TriaGrid
from pymor.grids.unstructured import UnstructuredTriangleGrid

from pymor.domaindescriptions.basic import RectDomain

from pymor.analyticalproblems.elliptic import EllipticProblem

from pymor.discretizers.elliptic import discretize_elliptic_cg

from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.parameters.base import Parameter

from pymor.parameters.spaces import CubicParameterSpace

from pymor.algorithms.ei import interpolate_operators

from domaintransformation.functions.basic import DomainTransformationFunction
from domaintransformation.analyticalproblems.elliptic_transformation import TransformationProblem
from domaintransformation.grids.domaintransformation import DomainTransformationTriaGrid
from domaintransformation.discretizers.elliptic import discretize_elliptic_cg

def parameter_mask(num_control_points, active=[]):
    assert isinstance(num_control_points, tuple)
    assert len(num_control_points) == 2

    K, L = num_control_points

    assert isinstance(active, list)
    assert len(active) <= K*L*2

    assert all(isinstance(a, tuple) for a in active)
    assert all(len(a) == 3 for a in active)
    assert all(isinstance(a, int) and isinstance(b, int) and isinstance(c, int) for a, b, c in active)
    assert all(x >= 0 and x < K and y >= 0 and y < L and z in [0, 1] for x, y, z in active)

    mask = np.zeros((K, L, 2), dtype=bool)
    for x,y,z in active:
        mask[x,y,z] = True

    return mask

class FFDRHSTransformation(DomainTransformationFunction):
    dim_domain = 2
    shape_range = 2

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

        super(FFDRHSTransformation, self).__init__(parameter_type)

        self.parameter_space = CubicParameterSpace(parameter_type, min, max, ranges)

    def evaluate(self, x, mu=None):
        assert mu is None or isinstance(mu, Parameter)
        if mu is None:
            return x
        else:
            mu = self._assemble_parameter(mu)
        return self.transform(x, mu)

    def transform(self, x, mu=None):
        assert mu is None or isinstance(mu, Parameter) or isinstance(mu, np.ndarray)
        if mu is None:
            return x
        if isinstance(mu, Parameter):
            mu = self._assemble_parameter(mu)
        P_mu = self.P_0 + mu

        x_ = self._bernstein2d_vec(self._psi(x))
        x_mu_sum = (x_[..., np.newaxis] * P_mu[np.newaxis, ...]).sum(axis=(1,2))

        return self._psi_inverse(x_mu_sum)

    def jacobian(self, x, mu=None):
        assert mu is None or isinstance(mu, Parameter) or isinstance(mu, np.ndarray)
        if mu is None:
            return np.eye(2).reshape((1, 2, 2)).repeat(x.shape[0], axis=0)
        if isinstance(mu, Parameter):
            mu = self._assemble_parameter(mu)
        x_ = self._bernstein2d_derivative_vec(self._psi(x))

        x_mu = x_[..., np.newaxis] * mu[np.newaxis, ...]
        x_sum = x_mu.sum(axis=(2, 3))
        x_sum += np.eye(2).reshape((1,2,2))
        x_sum = np.swapaxes(x_sum, 1, 2)

        psi = self._psi_jacobian()
        psi_inv = self._psi_inverse_jacobian()
        return np.einsum("ij,ejk,kl->eil", psi_inv, x_sum, psi)

    def jacobian_determinant(self, x, mu=None):
        J = self.jacobian(x, mu)
        det = np.linalg.det(J)
        return np.abs(det)

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

        # value of 0 is ensured later by multiplying by comb, which has a 0 at the corresponding entry
        SPECIAL = np.logical_or(k>K, k<0)
        if np.any(SPECIAL):
            k[SPECIAL] = 0

        X_ = (1. - X[..., np.newaxis])**(K-k) * X[..., np.newaxis]**k
        res = combs[np.newaxis, ...] * X_

        return res

    def _bernstein1d_derivative_vec(self, X, index):
        assert len(X.shape) == 1

        b_1 = self._bernstein1d_vec(X, index, -1, -1)
        b_2 = self._bernstein1d_vec(X, index, -1, 0)

        L = self.num_control_points[index]-1

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
        J_T = self.jacobian(x, mu)
        det = np.linalg.det(J_T)

        assert np.count_nonzero(det) == det.size, "Singular value"
        inv = np.linalg.inv(J_T)
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
        assert mu is None or isinstance(mu, Parameter) or isinstance(mu, np.ndarray)
        if mu is None:
            return np.eye(2).reshape((1, 2, 2)).repeat(x.shape[0], axis=0)
        if isinstance(mu, Parameter):
            mu = self._assemble_parameter(mu)
        x_ = self._bernstein2d_derivative_vec(self._psi(x))

        x_mu = x_[..., np.newaxis] * mu[np.newaxis, ...]
        x_sum = x_mu.sum(axis=(2, 3))
        x_sum += np.eye(2).reshape((1,2,2))
        x_sum = np.swapaxes(x_sum, 1, 2)

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

        # value of 0 is ensured later by multiplying by comb, which has a 0 at the corresponding entry
        SPECIAL = np.logical_or(k>K, k<0)
        if np.any(SPECIAL):
            k[SPECIAL] = 0

        X_ = (1. - X[..., np.newaxis])**(K-k) * X[..., np.newaxis]**k
        res = combs[np.newaxis, ...] * X_

        return res

    def _bernstein1d_derivative_vec(self, X, index):
        assert len(X.shape) == 1

        b_1 = self._bernstein1d_vec(X, index, -1, -1)
        b_2 = self._bernstein1d_vec(X, index, -1, 0)

        L = self.num_control_points[index]-1

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



domain = np.array([[0, 0], [1, 1]])
shift_min = -0.5
shift_max = 0.5
PROBLEM_NUMBER = 3
num_control_points = [(2, 2), (5, 3), (5, 3), (5, 3)][PROBLEM_NUMBER]
active = [[(0, 1, 0), (1, 1, 0)],
          [(4, 1, 0), (0, 2, 0)],
          [(0, 2, 1), (1, 2, 1), (2, 2, 1), (3, 2, 1), (4, 0, 0), (4, 1, 0), (4, 2, 0), (4, 2, 1)],
          [(1, 2, 1), (3, 2, 1), (4, 1, 0)]
         ][PROBLEM_NUMBER]
mask = parameter_mask(num_control_points, active)

param = [np.array([1.0, 1.0]),
         np.array([0.01, 0.03]),
         np.array([0.15, -0.15, 0.15, -0.15, 0.15, -0.25, 0.15, 0.15]),
         np.array([-0.15, 0.15, -0.15])]
mu = {'ffd': param[PROBLEM_NUMBER]}
mu = Parameter(mu)

problem = EllipticProblem(name="elliptic")

transformation_type = {'transformation': (2,2)}
transformation = FFDTransformation(RectDomain(domain), mask, "ffd", shift_min, shift_max)
rhs_transformation = FFDRHSTransformation(RectDomain(domain), mask, "ffd", shift_min, shift_max)
trafo_problem = TransformationProblem(problem, transformation, rhs_transformation)


def run(N, sample_size, ei_size, plot=False):
    print(N)
    num_intervals = (N, N)
    elliptic_grid = TriaGrid(num_intervals)
    trafo_grid = DomainTransformationTriaGrid(elliptic_grid, transformation)

    bi_t = AllDirichletBoundaryInfo(trafo_grid)

    discretization, _ = discretize_elliptic_cg(trafo_problem, grid=trafo_grid, boundary_info=bi_t)

    discretization_ei, ei_data = interpolate_operators(discretization, ['operator'],
                                                       discretization.parameter_space.sample_uniformly(sample_size),
                                                       error_norm=discretization.l2_norm,
                                                       target_error=1e-17,
                                                       max_interpolation_dofs=ei_size,
                                                       projection="orthogonal",
                                                       product=discretization.l2_product)

    #discretization.disable_caching()
    #discretization_ei.disable_caching()

    U = discretization.solve(mu)
    U_ei = discretization_ei.solve(mu)

    ERR = U - U_ei
    ERR_abs = ERR.copy()
    ERR_abs._array = np.abs(ERR_abs._array)

    if plot:
        discretization.visualize((U, U_ei, ERR, ERR_abs),
                                 mu=(mu,mu,mu,mu),
                                 legend=("FEM", "EI", "ERR", "abs(ERR)"),
                                 separate_colorbars=True,
                                 title="Grid {}x{}".format(*num_intervals))

    err_l2 = discretization.l2_norm(ERR)
    err_h1_semi = discretization.h1_semi_norm(ERR)
    err_h1 = discretization.h1_norm(ERR)
    return {"ERR_L2": err_l2, "ERR_H1_SEMI": err_h1_semi, "ERR_H1": err_h1}

MIN = 20
MAX = 200
ERRS_L2 = []
ERRS_H1_SEMI = []
ERRS_H1 = []
NS = []
SAMPLE_SIZE = 2
EI_SIZE = 100
GRID = 1

for N in [GRID]:
    NS.append(N)
    r = run(N, SAMPLE_SIZE, EI_SIZE, plot=True)
    ERRS_L2.append(r["ERR_L2"])
    ERRS_H1_SEMI.append(r["ERR_H1_SEMI"])
    ERRS_H1.append(r["ERR_H1"])



#print(NS)
#print(ERRS_L2)

#N_ = np.linspace(MIN,MAX,100)
#E_1 = 1.0/N_
#E_2 = 1.0/(N_**2)




#plt.plot(NS, ERRS_L2, label="ERR")
#plt.plot(N_, E_1, label="10E-1")
#plt.plot(N_, E_2, label="10E-2")
#plt.xlabel("Number of Intervals")
#plt.ylabel("L2 Error")
#plt.xscale('log')
#plt.yscale('log')
#plt.legend()
#plt.show()

#plt.plot(NS, ERRS_H1,label="ERR")
#plt.plot(N_, E_1, label="10E-1")
#plt.plot(N_, E_2, label="10E-2")
#plt.xlabel("Number of Intervals")
#plt.ylabel("H1 Error")
#plt.xscale('log')
#plt.yscale('log')
#plt.legend()
#plt.show()

#run(100, True, True)

