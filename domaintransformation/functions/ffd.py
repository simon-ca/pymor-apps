from pymor.domaindescriptions.basic import RectDomain
from pymor.parameters.spaces import CubicParameterSpace
from pymor.parameters.base import Parameter

from transformation import DomainTransformationFunction

import numpy as np


# symmetry constraints
CONSTRAINT_SAME = 1
CONSTRAINT_SAME_MIRROR = 2
CONSTRAINT_INVERSE = -1
CONSTRAINT_INVERSE_MIRROR = -2

# direction values
DIRECTION_NONE = 0
DIRECTION_X = 1
DIRECTION_Y = 2
DIRECTION_BOTH = DIRECTION_X + DIRECTION_Y
DIRECTIONS = [DIRECTION_NONE, DIRECTION_X, DIRECTION_Y, DIRECTION_BOTH]


class FreeFormDeformation(DomainTransformationFunction):

    dim_domain = 2
    shape_range = (2, 2)

    def __init__(self, domain, K, L, active, constraints, parameter_name, min, max, ranges=None):
        assert isinstance(domain, RectDomain)

        assert isinstance(K, int)
        assert isinstance(L, int)

        assert isinstance(active, list)
        assert all(isinstance(a, tuple) and len(a) == 3 and
                   0 <= a[0] <= K and 0 <= a[1] <= L and a[2] in DIRECTIONS for a in active)

        num_parameters = 0
        parameter_mask = np.zeros(shape=(K, L, 2), dtype=np.bool)
        for k, l, dir in active:
            if dir == DIRECTION_X:
                num_parameters += 1
                parameter_mask[k, l, 0] = True
            elif dir == DIRECTION_Y:
                num_parameters += 1
                parameter_mask[k, l, 1] = True
            elif dir == DIRECTION_BOTH:
                num_parameters += 2
                parameter_mask[k, l, :] = True
            elif dir == DIRECTION_NONE:
                pass
            else:
                raise NotImplementedError

        for c in constraints:
            if c:
                num_parameters -= 1

        assert isinstance(constraints, list)
        assert len(constraints) == len(active)
        assert all(c is None or
                   isinstance(c, tuple) and len(c) == 3 for c in constraints)

        # multiple references are not allowed
        assert all(c[0] < len(constraints) and constraints[c[0]] is None for c in constraints if c)

        assert isinstance(parameter_name, str)

        assert min is not None and max is not None and ranges is None

        self.domain = domain
        self.active = active
        self.constraints = constraints
        self.parameter_name = parameter_name

        self.parameter_mask = parameter_mask

        self.num_control_points = (K, L)

        parameter_shape = (num_parameters,)

        x = np.linspace(0, 1, K)
        y = np.linspace(0, 1, L)

        self.P_0 = np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))]).reshape((len(x), len(y), 2))

        parameter_type = {parameter_name: parameter_shape}

        super(FreeFormDeformation, self).__init__(parameter_type, name="FFD")

        self.parameter_space = CubicParameterSpace(parameter_type, min, max, ranges)

    def apply(self, x, mu=None):
        assert mu is None or isinstance(mu, Parameter)
        if mu is None:
            return x
        if isinstance(mu, Parameter):
            mu = self._assemble_parameter(mu)
        P_mu = self.P_0 + mu

        x_ = self._bernstein2d_vec(self._psi(x))
        x_mu_sum = (x_[..., np.newaxis] * P_mu[np.newaxis, ...]).sum(axis=(1, 2))

        return self._psi_inverse(x_mu_sum)

    def apply_inverse(self, x, mu=None):
        raise NotImplementedError

    def jacobian_inverse(self, x, mu=None):
        raise NotImplementedError

    def jacobian_determinant(self, x, mu=None):
        assert x.ndim in [2, 3]
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

        ll = domain[0, :]
        lr = np.array([domain[1, 0], domain[0, 1]])
        ul = np.array([domain[0, 0], domain[1, 1]])
        ur = domain[1, :]

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
        parameter = np.zeros_like(self.parameter_mask, dtype=np.float)

        # map non-constraints
        j = 0
        for i, c in enumerate(self.constraints):
            if c is None:
                k, l, dir = self.active[i]
                if dir == DIRECTION_X:
                    parameter[k, l, 0] = mu[j]
                    j += 1
                elif dir == DIRECTION_Y:
                    parameter[k, l, 1] = mu[j]
                    j += 1
                elif dir == DIRECTION_BOTH:
                    parameter[k, l, :] = mu[j]
                    j += 1
                i += 1
        # map constraints
        for i, c in enumerate(self.constraints):
            if c is not None:
                ref_point, x, y = c
                k_ref, l_ref, _ = self.active[ref_point]
                k_constraint, l_constraint, _ = self.active[i]
                if x == CONSTRAINT_SAME:
                    parameter[k_constraint, l_constraint, 0] = parameter[k_ref, l_ref, 0]
                elif x == CONSTRAINT_INVERSE:
                    parameter[k_constraint, l_constraint, 0] = -parameter[k_ref, l_ref, 0]
                elif x == CONSTRAINT_SAME_MIRROR:
                    parameter[k_constraint, l_constraint, 0] = parameter[k_ref, l_ref, 1]
                elif x == CONSTRAINT_INVERSE_MIRROR:
                    parameter[k_constraint, l_constraint, 0] = -parameter[k_ref, l_ref, 1]
                if y == CONSTRAINT_SAME:
                    parameter[k_constraint, l_constraint, 1] = parameter[k_ref, l_ref, 1]
                elif y == CONSTRAINT_INVERSE:
                    parameter[k_constraint, l_constraint, 1] = -parameter[k_ref, l_ref, 1]
                elif y == CONSTRAINT_SAME_MIRROR:
                    parameter[k_constraint, l_constraint, 1] = parameter[k_ref, l_ref, 0]
                elif y == CONSTRAINT_INVERSE_MIRROR:
                    parameter[k_constraint, l_constraint, 1] = -parameter[k_ref, l_ref, 0]

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
