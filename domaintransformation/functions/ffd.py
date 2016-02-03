from pymor.domaindescriptions.basic import RectDomain
from pymor.parameters.spaces import CubicParameterSpace
from pymor.parameters.base import Parameter

from transformation import DomainTransformationFunction

import numpy as np


class FreeFormDeformation(DomainTransformationFunction):

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

        super(FreeFormDeformation, self).__init__(parameter_type)

        self.parameter_space = CubicParameterSpace(parameter_type, min, max, ranges)

    def apply(self, x, mu=None):
        assert mu is None or isinstance(mu, Parameter)
        if mu is None:
            return x
        if isinstance(mu, Parameter):
            mu = self._assemble_parameter(mu)
        P_mu = self.P_0 + mu

        x_ = self._bernstein2d_vec(self._psi(x))
        x_mu_sum = (x_[..., np.newaxis] * P_mu[np.newaxis, ...]).sum(axis=(1,2))

        return self._psi_inverse(x_mu_sum)

    def apply_inverse(self, x, mu=None):
        raise NotImplementedError

    def jacobian_inverse(self, x, mu=None):
        raise NotImplementedError

    def jacobian_determinant(self, x, mu=None):
        assert x.ndim in [2,3]
        x_ = x.reshape((-1, self.dim_domain))
        res = np.linalg.det(self.jacobian(x_, mu))
        if x.ndim == 2:
            return res
        else:
            return res.reshape(x.shape[:2])

    def jacobian(self, x, mu=None):
        assert x.ndim in [2,3]
        x_ = x.reshape((-1, self.dim_domain))
        mu = self.parse_parameter(mu)
        assert mu is None or isinstance(mu, Parameter) or isinstance(mu, np.ndarray)
        if mu is None:
            return np.eye(2).reshape((1, 2, 2)).repeat(x.shape[0], axis=0)
        if isinstance(mu, Parameter):
            mu = self._assemble_parameter(mu)
        x_ = self._bernstein2d_derivative_vec(self._psi(x_))

        x_mu = x_[..., np.newaxis] * mu[np.newaxis, ...]
        x_sum = x_mu.sum(axis=(2, 3))
        x_sum += np.eye(2).reshape((1,2,2))
        x_sum = np.swapaxes(x_sum, 1, 2)

        psi = self._psi_jacobian()
        psi_inv = self._psi_inverse_jacobian()
        res = np.einsum("ij,ejk,kl->eil", psi_inv, x_sum, psi)
        return res.reshape(x.shape[:x.ndim-1]+self.shape_range)

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
            m = self._assemble_parameter(m)
            P_m = self.P_0 + m
            box_transformed.append(P_m)
        box_transformed = np.array(box_transformed)

        max = box_transformed.max(axis=(0,1,2))
        min = box_transformed.min(axis=(0,1,2))

        return np.array([min, max])


    def _assemble_parameter(self, mu):
        mu = self.parse_parameter(mu)
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



