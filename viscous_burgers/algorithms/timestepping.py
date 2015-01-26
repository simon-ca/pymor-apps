# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse
from pymor.la.interfaces import VectorArrayInterface
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator


def imex_euler(A, D, F, U0, t0, t1, nt, mu):
    assert isinstance(A, OperatorInterface)
    assert isinstance(D, OperatorInterface)
    assert A.source == A.range == D.source == D.range
    A_time_dep = A.parameter_type and '_t' in A.parameter_type
    D_time_dep = D.parameter_type and '_t' in D.parameter_type
    if not D_time_dep:
        D = D.assemble(mu)

    assert isinstance(F, (OperatorInterface, VectorArrayInterface))
    if isinstance(F, OperatorInterface):
        assert F.range.dim == 1
        assert F.source == A.source
        F_time_dep = F.parameter_type and '_t' in F.parameter_type
        if not F_time_dep:
            F = F.as_vector(mu)
    else:
        assert len(F) == 1
        assert F.dim in A.source
        F_time_dep = False

    assert isinstance(U0, VectorArrayInterface)
    assert len(U0) == 1
    assert U0 in A.source

    dt = (t1 - t0) / nt
    R = A.source.empty(reserve=nt+1)
    R.append(U0)

    if hasattr(D, 'sparse') and D.sparse:
        I = NumpyMatrixOperator(scipy.sparse.eye(A.source.dim, A.source.dim))
    else:
        I = NumpyMatrixOperator(np.eye(A.source.dim))
        
    if not D_time_dep:
        dt_D = I + D * dt
    if not F_time_dep:
        dt_F = F * dt

    t = t0
    U = U0.copy()

    for n in xrange(nt):
        t += dt
        mu['_t'] = t
        if D_time_dep:
            dt_D = I + D.assemble(mu) * dt
        if F_time_dep:
            dt_F = F.assemble(mu) * dt
        U = dt_D.apply_inverse(U + dt_F - A.apply(U, mu=mu) * dt)
        R.append(U)

    return R

