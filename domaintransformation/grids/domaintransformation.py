from __future__ import absolute_import, division, print_function

from pymor.grids.tria import TriaGrid
from pymor.core.cache import cached
from pymor.grids.interfaces import AffineGridWithOrthogonalCentersInterface
from pymor.grids.referenceelements import triangle

from pymor.parameters.base import Parameter

from domaintransformation.functions.basic import DomainTransformationFunction

import numpy as np


class DomainTransformationTriaGrid(TriaGrid):
    """Transformed TriaGrid

    Parameters
    ----------
    grid
        |TriaGrid|
    transformation
    """

    dim = 2
    dim_outer = 2
    reference_element = triangle

    def __init__(self, grid, transformation):
        assert isinstance(grid, TriaGrid)
        assert isinstance(transformation, DomainTransformationFunction)

        self.grid = grid
        self.transformation = transformation

        num_intervals = grid.num_intervals
        domain = grid.domain
        id_bottom_top = grid.identify_bottom_top
        id_left_right = grid.identify_left_right
        super(DomainTransformationTriaGrid, self).__init__(num_intervals, domain, id_left_right, id_bottom_top)

    def bounding_box(self, mu=None):
        assert mu is None or isinstance(mu, Parameter) \
               or isinstance(mu, tuple) and all(isinstance(m, Parameter) for m in mu)
        if mu is None:
            return self.domain
        else:
            return self.transformation.bounding_box(self.domain, mu)

    def centers(self, codim, mu=None):
        assert mu is None or isinstance(mu, Parameter)
        centers = self.grid.centers(codim)
        if mu is None:
            return centers
        else:
            return self.transformation.transform(centers, mu)

    def embeddings(self, codim=0, mu=None):
        assert mu is None or isinstance(mu, Parameter)
        A, B = self.grid.embeddings(codim)
        if mu is None:
            return A, B
        else:
            B_transformed = self.transformation.transform(B, mu)

            if codim == 2:
                # A is empty for points
                A_transformed = A
            else:
                A_transformed = self.transformation.transform(A, mu)

            return A_transformed, B_transformed

    def visualize(self, U, codim=2, **kwargs):
        """Visualize scalar data associated to the grid as a patch plot.
        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case a subplot is created for each entry of the tuple. The
            lengths of all arrays have to agree.
        mu
            Parameter
        codim
            The codimension of the entities the data in `U` is attached to (either 0 or 2).
        kwargs
            See :func:`~pymor.gui.qt.visualize_patch`
        """
        from domaintransformation.gui.qt import visualize_patch
        from pymor.vectorarrays.numpy import NumpyVectorArray
        #if not isinstance(U, NumpyVectorArray):
        #    U = NumpyVectorArray(U, copy=False)

        mu = kwargs.pop('mu', None)
        bounding_box = self.bounding_box(mu)
        visualize_patch(self, U, mu, codim=codim, bounding_box=bounding_box, **kwargs)
