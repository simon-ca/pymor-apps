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

from domaintransformation.functions.basic import DomainTransformationFunction
from domaintransformation.analyticalproblems.elliptic_transformation import AffineTransformationProblem, TransformationProblem
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
    assert all(0 <= x < K and 0 <= y < L and z in [0, 1] for x, y, z in active)

    mask = np.zeros((K, L, 2), dtype=bool)
    for x,y,z in active:
        mask[x,y,z] = True

    return mask

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

    def diffusion(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        det = np.linalg.det(mu)
        assert not det == 0

        A_inv = np.linalg.inv(mu)

        res = A_inv.dot(A_inv.T)
        # FIXME det is not handled on RHS
        res *=np.abs(det)

        return res.reshape((1, 2, 2)).repeat(x.shape[0], axis=0)

    def transform(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        assert len(x.shape) == 2
        return np.einsum("ij,ej->ei", mu, x)

    def jacobian(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        res = mu.reshape((1, 2, 2)).repeat(x.shape[0], axis=0)
        return res

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

    def jacobian_determinant(self, x, mu=None):
        J = self.jacobian(x, mu)
        return np.abs(np.linalg.det(J))

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
        # FIXME det is not handled on RHS
        det = np.ones_like(det)
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



def run(trafo_number):
    assert isinstance(trafo_number, int)
    assert 0 <= trafo_number <= 5, "trafo must be between 0 and 5"

    check = True
    triplot = False
    plot = True

    num_intervals = (20, 20)
    domain = np.array([[0, 0], [1, 1]])
    affine_min = -2.0
    affine_max = 2.0
    ffd_min = -2.0
    ffd_max = 2.0

    num_control_points = (2, 2)
    active = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    mask = parameter_mask(num_control_points, active)

    affine = {0: np.array([[1.0, 0.0], [0.0, 1.0]]),
              1: np.array([[2.0, 0.0], [0.0, 2.0]]),
              2: np.array([[1.0, 1.0], [0.0, 1.0]]),
              3: np.array([[-1.0, 0.0], [0.0, -1.0]]),
              4: np.array([[0.0, -1.0], [1.0, 0.0]]),
              5: np.array([[1.0/np.sqrt(2), -1.0/np.sqrt(2)], [1.0/np.sqrt(2), 1.0/np.sqrt(2)]])
              }
    ffd = {0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
           1: np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
           2: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
           3: np.array([0.0, 0.0, 0.0, -2.0, -2.0, 0.0, -2.0, -2.0]),
           4: np.array([0.0, 0.0, -1.0, -1.0, -1.0, 1.0, -2.0, 0.0]),
           5: np.array([0.0, 0.0, -1.0/np.sqrt(2), 1.0/np.sqrt(2)-1.0, 1.0/np.sqrt(2)-1.0, 1.0/np.sqrt(2), -1.0, np.sqrt(2)-1.0]),
           }

    mu_affine = {'transformation': affine[trafo_number]}
    mu_ffd = {'ffd': ffd[trafo_number]}

    mu_affine = Parameter(mu_affine)
    mu_ffd = Parameter(mu_ffd)

    problem_elliptic = EllipticProblem(name='elliptic')

    trafo_type_affine = {'transformation': (2,2)}
    trafo_affine = AffineTransformation(trafo_type_affine, affine_min, affine_max)
    problem_trafo_affine = AffineTransformationProblem(problem_elliptic, trafo_affine)

    trafo_ffd = FFDTransformation(RectDomain(domain), mask, "ffd", ffd_min, ffd_max)
    rhs_ffd = FFDRHSTransformation(RectDomain(domain), mask, "ffd", ffd_min, ffd_max)
    problem_trafo_ffd = TransformationProblem(problem_elliptic, trafo_ffd, rhs_ffd)

    grid_elliptic = TriaGrid(num_intervals, domain)
    grid_trafo_affine = DomainTransformationTriaGrid(grid_elliptic, trafo_affine)
    grid_trafo_ffd = DomainTransformationTriaGrid(grid_elliptic, trafo_ffd)

    # unstructured grid
    vertices = grid_elliptic.centers(2)
    vertices_affine = grid_trafo_affine.centers(2, mu_affine)
    vertices_ffd = grid_trafo_ffd.centers(2, mu_ffd)
    faces = grid_elliptic.subentities(0,2)

    grid_unstr_affine = UnstructuredTriangleGrid(vertices_affine, faces)
    grid_unstr_ffd = UnstructuredTriangleGrid(vertices_ffd, faces)

    bi_affine = AllDirichletBoundaryInfo(grid_trafo_affine)
    bi_ffd = AllDirichletBoundaryInfo(grid_trafo_ffd)
    bi_unstr_affine = AllDirichletBoundaryInfo(grid_unstr_affine)
    bi_unstr_ffd = AllDirichletBoundaryInfo(grid_unstr_ffd)

    discretization_affine, _ = discretize_elliptic_cg(problem_trafo_affine, grid=grid_trafo_affine, boundary_info=bi_affine)
    discretization_ffd, _ = discretize_elliptic_cg(problem_trafo_ffd, grid=grid_trafo_ffd, boundary_info=bi_ffd)
    discretization_unstr_affine, _ = discretize_elliptic_cg(problem_elliptic, grid=grid_unstr_affine, boundary_info=bi_unstr_affine)
    discretization_unstr_ffd, _ = discretize_elliptic_cg(problem_elliptic, grid=grid_unstr_ffd, boundary_info=bi_unstr_ffd)

    discretization_affine.disable_caching()
    discretization_ffd.disable_caching()
    discretization_unstr_affine.disable_caching()
    discretization_unstr_ffd.disable_caching()

    U_affine = discretization_affine.solve(mu_affine)
    U_ffd = discretization_ffd.solve(mu_ffd)
    U_unstr_affine = discretization_unstr_affine.solve()
    U_unstr_ffd = discretization_unstr_ffd.solve()
    ERR_affine = U_affine - U_unstr_affine
    ERR_ffd = U_ffd - U_unstr_ffd
    ERR_affine_abs = ERR_affine.copy()
    ERR_affine_abs._array = np.abs(ERR_affine_abs._array)
    ERR_ffd_abs = ERR_ffd.copy()
    ERR_ffd_abs._array = np.abs(ERR_ffd_abs._array)

    if triplot:
        fig = plt.figure()
        fig.add_subplot(211)
        tri_affine = plt.triplot(vertices_affine[..., 0], vertices_affine[..., 1], faces)
        fig.add_subplot(212)
        tri_ffd = plt.triplot(vertices_ffd[..., 0], vertices_ffd[..., 1], faces)
        plt.show()

    if check:

        def check_vector_array(a, b):
            return np.allclose(a._array, b._array)
        #check vertices
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        X_affine = trafo_affine.transform(X, mu_affine)
        X_ffd = trafo_ffd.transform(X, mu_ffd)

        assert np.allclose(X_affine, X_ffd)

        #check grids
        assert np.allclose(vertices_affine, vertices_ffd)

        #check affine <-> unstructured affine
        #assert check_vector_array(U_affine, U_unstr_affine), "Error in affine {}".format(trafo_number)
        #check ffd <-> unstructured ffd
        #assert check_vector_array(U_ffd, U_unstr_ffd)

        #check jacobian is constant

        A_Jac_affine = trafo_affine.jacobian(vertices, mu_affine)
        A_Jac_ffd = trafo_ffd.jacobian(vertices, mu_ffd)
        assert all(np.allclose(A_Jac_affine[0,...], A_Jac_affine[i,...]) for i in range(A_Jac_affine.shape[0]))
        assert all(np.allclose(A_Jac_ffd[0,...], A_Jac_ffd[i,...]) for i in range(A_Jac_ffd.shape[0]))

        assert np.allclose(A_Jac_affine, A_Jac_ffd), "Jacobians differ in transformation {}".format(trafo_number)

    if plot:
        discretization_affine.visualize((U_affine, U_unstr_affine, ERR_affine, ERR_affine_abs),
                                        mu=(mu_affine, mu_affine, mu_affine, mu_affine),
                                        legend=("Trafo", "Unstructured", "ERR", "abs(ERR)"),
                                        separate_colorbars=True,
                                        title="Affine [{}]".format(trafo_number))

        discretization_ffd.visualize((U_ffd, U_unstr_ffd, ERR_ffd, ERR_ffd_abs),
                                        mu=(mu_ffd, mu_ffd, mu_ffd, mu_ffd),
                                        legend=("Trafo", "Unstructured", "ERR", "abs(ERR)"),
                                        separate_colorbars=True,
                                        title="FFD [{}]".format(trafo_number))


    #plt.triplot(vertices_affine[..., 0], vertices_affine[..., 1], faces)
    #plt.show()

    #plt.triplot(vertices_ffd[..., 0], vertices_ffd[..., 1], faces)
    #plt.show()

    #discretization_affine.visualize((U_affine, U_unstr_affine, ERR_affine, ERR_affine_abs),
    #                                mu=(mu_affine, mu_affine, mu_affine, mu_affine),
    #                                legend=("Trafo", "Unstructured", "ERR", "abs(ERR)"),
    #                                separate_colorbars=True,
    #                                title="Affine [{}]".format(trafo_number))

    #discretization_ffd.visualize((U_ffd, U_unstr_ffd, ERR_ffd, ERR_ffd_abs),
    #                                mu=(mu_ffd, mu_ffd, mu_ffd, mu_ffd),
    #                                legend=("Trafo", "Unstructured", "ERR", "abs(ERR)"),
    #                                separate_colorbars=True,
    #                                title="FFD [{}]".format(trafo_number))


for i in set([0,1,2,3,4,5]) - set([0,2,3,4,5]):
    #if i==1:
    #    continue
    pass
    run(i)


def scale():
    num_intervals = (200, 200)
    domain_1 = np.array([[0, 0], [1, 1]])
    domain_2 = np.array([[0, 0], [2, 2]])
    affine_min = -2.0
    affine_max = 2.0

    mu = {'transformation': np.array([[2.0, 0.0], [0.0, 2.0]])}
    mu = Parameter(mu)

    problem_elliptic = EllipticProblem(name='elliptic')

    trafo_type_affine = {'transformation': (2,2)}
    trafo_affine = AffineTransformation(trafo_type_affine, affine_min, affine_max)
    problem_trafo_1 = AffineTransformationProblem(problem_elliptic, trafo_affine)

    grid_1 = TriaGrid(num_intervals, domain_1)
    grid_2 = TriaGrid(num_intervals, domain_2)
    grid_trafo_1 = DomainTransformationTriaGrid(grid_1, trafo_affine)

    bi_2 = AllDirichletBoundaryInfo(grid_2)
    bi_trafo_1 = AllDirichletBoundaryInfo(grid_trafo_1)

    discretization_2, _ = discretize_elliptic_cg(problem_elliptic, grid=grid_2, boundary_info=bi_2)
    discretization_trafo_1, _ = discretize_elliptic_cg(problem_trafo_1, grid=grid_trafo_1, boundary_info=bi_trafo_1)

    discretization_2.disable_caching()
    discretization_trafo_1.disable_caching()

    U_trafo_1 = discretization_trafo_1.solve(mu)
    U_2 = discretization_2.solve()

    discretization_trafo_1.visualize((U_2, U_trafo_1),
                                        mu=(mu, mu),
                                        legend=("FEM", "Trafo"),
                                        separate_colorbars=True,
                                        title="Scale")

    grid_1.visualize(U_2)






#plt.triplot(vertices[..., 0], vertices[..., 1], faces)
#plt.show()



