from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import coo_matrix

from pymor.grids.referenceelements import triangle, line, square
from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class L2ProductFunctionalP1(NumpyMatrixBasedOperator):
    """|Functional| representing the scalar product with an L2-|Function| for linear finite elements.

    Boundary treatment can be performed by providing `boundary_info` and `dirichlet_data`,
    in which case the DOFs corresponding to Dirichlet boundaries are set to the values
    provided by `dirichlet_data`. Neumann boundaries are handled by providing a
    `neumann_data` function, Robin boundaries by providing a `robin_data` tuple.

    The current implementation works in one and two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the scalar product.
    boundary_info
        |BoundaryInfo| determining the Dirichlet and Neumann boundaries or `None`.
        If `None`, no boundary treatment is performed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values. If `None`,
        constant-zero boundary is assumed.
    neumann_data
        |Function| providing the Neumann boundary values. If `None`,
        constant-zero is assumed.
    robin_data
        Tuple of two |Functions| providing the Robin parameter and boundary values, see `RobinBoundaryOperator`.
        If `None`, constant-zero for both functions is assumed.
    order
        Order of the Gauss quadrature to use for numerical integration.
    name
        The name of the functional.
    clear_dirichlet
        If True clears all dirichlet dofs
    clear non_dirichlet
        if True clears all non-dirichlet dofs
    """

    sparse = False
    range = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, name=None, clear_dirichlet=False, clear_non_dirichlet=False):
        assert grid.reference_element(0) in {line, triangle}
        assert function.shape_range == tuple()
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.name = name
        self.clear_dirichlet = clear_dirichlet
        self.clear_non_dirichlet = clear_non_dirichlet
        self.build_parameter_type(inherits=(function, dirichlet_data, neumann_data))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points)
        F = self.function(g.quadrature_points(0, order=self.order), mu=mu)

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=self.order)
        if g.dim == 1:
            SF = np.array((1 - q[..., 0], q[..., 0]))
        elif g.dim == 2:
            SF = np.array(((1 - np.sum(q, axis=-1)),
                           q[..., 0],
                           q[..., 1]))
        else:
            raise NotImplementedError

        if self.clear_non_dirichlet:
            F = 0.0 * F

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFs
        # FIXME This implementation is horrible, find a better way!
        SF_I = g.subentities(0, g.dim).ravel()
        I = np.array(coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).todense()).ravel()

        # neumann boundary treatment
        if bi is not None and bi.has_neumann and self.neumann_data is not None:
            NI = bi.neumann_boundaries(1)
            if g.dim == 1:
                I[NI] -= self.neumann_data(g.centers(1)[NI])
            else:
                F = -self.neumann_data(g.quadrature_points(1, order=self.order)[NI], mu=mu)
                q, w = line.quadrature(order=self.order)
                SF = np.squeeze(np.array([1 - q, q]))
                SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[NI], w).ravel()
                SF_I = g.subentities(1, 2)[NI].ravel()
                I += np.array(coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim)))
                                        .todense()).ravel()

        # robin boundary treatment
        if bi is not None and bi.has_robin and self.robin_data is not None:
            RI = bi.robin_boundaries(1)
            if g.dim == 1:
                xref = g.centers(1)[RI]
                I[RI] += (self.robin_data[0](xref) * self.robin_data[1](xref))
            else:
                xref = g.quadrature_points(1, order=self.order)[RI]
                F = (self.robin_data[0](xref, mu=mu) * self.robin_data[1](xref, mu=mu))
                q, w = line.quadrature(order=self.order)
                SF = np.squeeze(np.array([1 - q, q]))
                SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[RI], w).ravel()
                SF_I = g.subentities(1, 2)[RI].ravel()
                I += np.array(coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim)))
                                        .todense()).ravel()

        if bi is not None and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            if self.dirichlet_data is not None and not self.clear_dirichlet:
                I[DI] = self.dirichlet_data(g.centers(g.dim)[DI], mu=mu)
            else:
                I[DI] = 0

        return I.reshape((1, -1))


class L2ProductFunctionalQ1(NumpyMatrixBasedOperator):
    """|Functional| representing the scalar product with an L2-|Function| for bilinear finite elements.

    Boundary treatment can be performed by providing `boundary_info` and `dirichlet_data`,
    in which case the DOFs corresponding to Dirichlet boundaries are set to the values
    provided by `dirichlet_data`. Neumann boundaries are handled by providing a
    `neumann_data` function, Robin boundaries by providing a `robin_data` tuple.

    The current implementation works in two dimensions, but can be trivially
    extended to arbitrary dimensions.

    Parameters
    ----------
    grid
        |Grid| for which to assemble the functional.
    function
        The |Function| with which to take the scalar product.
    boundary_info
        |BoundaryInfo| determining the Dirichlet boundaries or `None`.
        If `None`, no boundary treatment is performed.
    dirichlet_data
        |Function| providing the Dirichlet boundary values. If `None`,
        constant-zero boundary is assumed.
    neumann_data
        |Function| providing the Neumann boundary values. If `None`,
        constant-zero is assumed.
    robin_data
        Tuple of two |Functions| providing the Robin parameter and boundary values, see `RobinBoundaryOperator`.
        If `None`, constant-zero for both functions is assumed.
    order
        Order of the Gauss quadrature to use for numerical integration.
    name
        The name of the functional.
    clear_dirichlet
        If True clears all dirichlet dofs
    clear non_dirichlet
        if True clears all non-dirichlet dofs
    """

    sparse = False
    range = NumpyVectorSpace(1)

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, neumann_data=None, robin_data=None,
                 order=2, name=None, clear_dirichlet=False, clear_non_dirichlet=False):
        assert grid.reference_element(0) in {square}
        assert function.shape_range == tuple()
        self.source = NumpyVectorSpace(grid.size(grid.dim))
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.order = order
        self.name = name
        self.clear_dirichlet = clear_dirichlet
        self.clear_non_dirichlet = clear_non_dirichlet
        self.build_parameter_type(inherits=(function, dirichlet_data))

    def _assemble(self, mu=None):
        g = self.grid
        bi = self.boundary_info

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points)
        F = self.function(g.quadrature_points(0, order=self.order), mu=mu)

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=self.order)
        if g.dim == 2:
            SF = np.array(((1 - q[..., 0]) * (1 - q[..., 1]),
                           (1 - q[..., 1]) * (q[..., 0]),
                           (q[..., 0]) * (q[..., 1]),
                           (q[..., 1]) * (1 - q[..., 0])))
        else:
            raise NotImplementedError

        if self.clear_non_dirichlet:
            F = 0.0 * F

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFs
        # FIXME This implementation is horrible, find a better way!
        SF_I = g.subentities(0, g.dim).ravel()
        I = np.array(coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim))).todense()).ravel()

        # neumann boundary treatment
        if bi is not None and bi.has_neumann and self.neumann_data is not None:
            NI = bi.neumann_boundaries(1)
            F = -self.neumann_data(g.quadrature_points(1, order=self.order)[NI], mu=mu)
            q, w = line.quadrature(order=self.order)
            SF = np.squeeze(np.array([1 - q, q]))
            SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[NI], w).ravel()
            SF_I = g.subentities(1, 2)[NI].ravel()
            I += np.array(coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim)))
                                    .todense()).ravel()

        if bi is not None and bi.has_robin and self.robin_data is not None:
            RI = bi.robin_boundaries(1)
            xref = g.quadrature_points(1, order=self.order)[RI]
            F = self.robin_data[0](xref, mu=mu) * self.robin_data[1](xref, mu=mu)
            q, w = line.quadrature(order=self.order)
            SF = np.squeeze(np.array([1 - q, q]))
            SF_INTS = np.einsum('ei,pi,e,i->ep', F, SF, g.integration_elements(1)[RI], w).ravel()
            SF_I = g.subentities(1, 2)[RI].ravel()
            I += np.array(coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim)))
                                    .todense()).ravel()

        if bi is not None and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            if self.dirichlet_data is not None and not self.clear_dirichlet:
                I[DI] = self.dirichlet_data(g.centers(g.dim)[DI], mu=mu)
            else:
                I[DI] = 0

        return I.reshape((1, -1))
