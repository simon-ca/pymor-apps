from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, vstack

from pymor.functions.interfaces import FunctionInterface
from pymor.functions.basic import GenericFunction
from pymor.grids.referenceelements import triangle, line
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.operators.cg import L2ProductFunctionalP1

class AdvectionOperatorP1(NumpyMatrixBasedOperator):
    """
    An advection operator. Stacks x_i components.
    """

    def __init__(self, grid, boundary_info, advection_function=None, dirichlet_clear_rows=True, name=None):
        self.grid = grid
        self.boundary_info = boundary_info
        self.advection_function = advection_function
        self.dirichlet_clear_rows = dirichlet_clear_rows
        self.name = name
        # TODO ???
        self.range = NumpyVectorSpace(2*grid.size(grid.dim))
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        if advection_function is not None:
            self.build_parameter_type(inherits=(advection_function,))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        if g.dim == 1:
            raise NotImplementedError
        elif g.dim == 2:
            num_local_psf = 3
            num_local_vsf = 3
            num_global_psf = g.size(g.dim)
            num_global_vsf = g.size(g.dim)
            PN = g.subentities(0, g.dim)
            VN = g.subentities(0, g.dim)
        else:
            raise NotImplementedError

        q, w = g.reference_element.quadrature(order=2)

        PSF = P1ShapeFunctions(g.dim)(q)
        VSF_GRAD = P1ShapeFunctionGradients(g.dim)(q)
        del q


        #transform gradients
        VSF_GRADS = np.einsum('eij,pj->epi', g.jacobian_inverse_transposed(0), VSF_GRAD)
        del VSF_GRAD

        # evaluate advection_function

        # calculate scalar products
        if self.advection_function is not None and self.advection_function.shape_range == tuple():
            A = self.advection_function(self.grid.centers(0), mu=mu)
            INTS = np.einsum('pq,evi,e,q,e->evpi', PSF, VSF_GRADS, g.integration_elements(0), w, A)
            del A
        elif self.advection_function is not None:
            A = self.advection_function(self.grid.centers(0), mu=mu)
            INTS = np.einsum('pq,evi,e,q, eji->evpj', PSF, VSF_GRADS, g.integration_elements(0), w, A)
            del A
        else:
            INTS = np.einsum('pq,evi,e,q->evpi', PSF, VSF_GRADS, g.integration_elements(0), w)
        del PSF, VSF_GRADS, w

        INTS_X = INTS[..., 0].ravel()
        INTS_Y = INTS[..., 1].ravel()
        del INTS

        SF_I0 = np.repeat(VN, num_local_psf, axis=1).ravel()
        SF_I1 = np.tile(PN, [1, num_local_vsf]).ravel()
        del PN, VN

        if bi is not None and bi.has_dirichlet:
            # set whole row to zero on boundary nodes
            # d_m = bi.dirichlet_mask(g.dim)
            if self.dirichlet_clear_rows:
                INTS_X = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, INTS_X)
                INTS_Y = np.where(bi.dirichlet_mask(g.dim)[SF_I0], 0, INTS_Y)

        B_X = coo_matrix((INTS_X, (SF_I0, SF_I1)), shape=(num_global_vsf, num_global_psf))
        B_Y = coo_matrix((INTS_Y, (SF_I0, SF_I1)), shape=(num_global_vsf, num_global_psf))
        del SF_I0, SF_I1, INTS_X, INTS_Y

        return vstack((csc_matrix(B_X).copy(), csc_matrix(B_Y).copy()))
