from __future__ import absolute_import, division, print_function

from pymor.functions.basic import FunctionBase

import numpy as np


class DomainTransformationFunction(FunctionBase):

    def __init__(self, parameter_type, name=None):
        self.build_parameter_type(parameter_type, local_global=True)
        self.name = name

    def evaluate(self, x, mu=None):
        return self.apply(x, mu)

    def apply(self, x, mu=None):
        raise NotImplementedError

    def apply_inverse(self, x, mu=None):
        raise NotImplementedError

    def jacobian(self, x, mu=None):
        raise NotImplementedError

    def jacobian_inverse(self, x, mu=None):
        raise NotImplementedError

    def jacobian_determinant(self, x, mu=None):
        raise NotImplementedError

    def bounding_box(self, domain, mu):
        raise NotImplementedError

class DiffusionTransformation(FunctionBase):

    def __init__(self, function):
        assert isinstance(function, DomainTransformationFunction)

        # todo only function from R^2 to R^(2x2) for now
        assert function.dim_domain == 2
        assert function.shape_range == (2,2)

        self.dim_domain = function.dim_domain
        self.shape_range = function.shape_range

        self.function = function
        self.build_parameter_type(inherits=[function])

    def evaluate(self, x, mu=None):
        mu = self.parse_parameter(mu)
        jac = self.function.jacobian(x, mu)
        jac_inv = np.linalg.inv(jac)

        assert jac_inv.shape == (x.shape[0],) + self.shape_range

        jac_inv_t = jac_inv.swapaxes(-1, -2)

        det = self.function.jacobian_determinant(x, mu)
        det = np.abs(det)
        assert det.shape == (x.shape[0],)

        assert jac_inv.ndim == 3
        assert det.ndim == 1

        res = np.einsum('eij,ejk,e->eik', jac_inv, jac_inv_t, det)
        return res

        # todo multiplication in last 2 dimensions
        raise NotImplementedError
        return None

class JacobianDeterminantTransformation(FunctionBase):

    def __init__(self, function):
        assert isinstance(function, DomainTransformationFunction)

        assert function.dim_domain == 2
        assert function.shape_range == (2,2)

        # todo only function from R^2 to R^(2x2) for now
        self.dim_domain = function.dim_domain
        self.shape_range = tuple()

        self.function = function
        self.build_parameter_type(inherits=[function])

    def evaluate(self, x, mu=None):
        # todo fix this
        #x_ = x.reshape((-1,2))
        mu = self.parse_parameter(mu)
        det = self.function.jacobian_determinant(x, mu)
        det = np.abs(det)

        #det = det.reshape((x.shape[0],-1))
        assert det.shape == x.shape[:2]

        return det
