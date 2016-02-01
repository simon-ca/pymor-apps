from pymor.functions.interfaces import FunctionInterface
from domaintransformation.functions.ei import EmpiricalInterpolatedFunction

from pymor.parameters.base import Parameter

from pymor.grids.interfaces import AffineGridWithOrthogonalCentersInterface

import numpy as np

def ei_greedy_function(U, error_norm=None, target_error=None, max_interpolation_dofs=None):
    assert isinstance(U, np.ndarray)
    assert len(U.shape) == 2 # mu x X

    interpolation_dofs = np.zeros((0,), dtype=np.int32)

    collateral_basis = []
    max_errs = []
    triangularity_errs = []

    ERR = U

    while True:
        # Calculate maximum norm
        errs = np.max(np.abs(ERR), axis=1)

        assert len(errs.shape) == 1
        assert errs.shape == (ERR.shape[0],)

        max_err_ind = np.argmax(errs)  # mu_m
        max_err = errs[max_err_ind]

        if len(interpolation_dofs) >= max_interpolation_dofs:
            print("Maximum number interpolation dofs")
            break
        if target_error is not None and max_err <= target_error:
            print("Target Error")
            break

        #EIM
        new_vec = U[max_err_ind,...].copy()
        assert len(new_vec.shape) == len(U.shape) - 1
        assert new_vec.shape == U.shape[1:]

        new_dof_index = np.argmax(np.abs(new_vec))

        if new_dof_index in interpolation_dofs:
            print("DOF taken twice")
            break

        new_dof_value = new_vec[new_dof_index]  # x_m

        if new_dof_value == 0.:
            print("Error Zero")
            break

        new_vec *= 1 / new_dof_value  # q_m

        interpolation_dofs = np.hstack((interpolation_dofs, new_dof_index))
        #interpolation_points = np.hstack(interpolation_points, )
        #collateral_basis = np.hstack((collateral_basis, new_vec))
        collateral_basis.append(new_vec)
        max_errs.append(max_err)

        # update U and ERR
        new_dof_values = U[:, new_dof_index]
        U -= np.outer(new_dof_values, new_vec)
        #U -= new_dof_values[...,np.newaxis] * new_vec[np.newaxis,...]

    collateral_basis = np.array(collateral_basis)

    interpolation_matrix = collateral_basis[:, interpolation_dofs].T
    triangularity_errors = np.abs(interpolation_matrix - np.tril(interpolation_matrix))
    for d in range(1, len(interpolation_matrix)+1):
        triangularity_errs.append(np.max(triangularity_errors[:d, :d]))

    data = {'errors': max_errs, 'triangularity_errors': triangularity_errs}

    return interpolation_dofs, collateral_basis, data


def ei_greedy_function_COPY(U, error_norm=None, target_error=None, max_interpolation_dofs=None):
    assert isinstance(U, np.ndarray)

    shape_range = U.ndim-1
    assert shape_range > 0

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    interpolation_dofs_tuple = []
    collateral_basis = []
    max_errs = []
    triangularity_errs = []

    ERR = U

    while True:
        # Calculate maximum norm
        ERR_abs = np.abs(ERR)
        ax = tuple(range(1, ERR.ndim))
        errs = np.max(ERR_abs, axis=ax)

        assert errs.ndim == 1
        assert errs.shape == (ERR.shape[0],)

        max_err_ind = np.argmax(errs)  # mu_m
        max_err = errs[max_err_ind]

        if len(interpolation_dofs) >= max_interpolation_dofs:
            print("Maximum number interpolation dofs")
            break
        if target_error is not None and max_err <= target_error:
            print("Target Error")
            break

        #EIM
        new_vec = U[max_err_ind,...].copy()
        assert new_vec.ndim == U.ndim - 1
        assert new_vec.shape == U.shape[1:]

        new_dof_index = np.argmax(np.abs(new_vec))

        # new_dof_indexed is a flattened index
        new_dof_index_tuple = np.unravel_index(new_dof_index, new_vec.shape)

        if new_dof_index in interpolation_dofs:
            print("DOF taken twice")
            break

        new_dof_value = new_vec[new_dof_index_tuple]  # x_m

        if new_dof_value == 0.:
            print("Error Zero")
            break

        new_vec *= 1 / new_dof_value  # q_m

        interpolation_dofs = np.hstack((interpolation_dofs, new_dof_index))
        interpolation_dofs_tuple.append(new_dof_index_tuple)
        #interpolation_points = np.hstack(interpolation_points, )
        #collateral_basis = np.hstack((collateral_basis, new_vec))
        collateral_basis.append(new_vec)
        max_errs.append(max_err)

        # update U and ERR
        index = (slice(None),) + new_dof_index_tuple
        assert len(index) == U.ndim

        new_dof_values = U[index]
        assert new_dof_values.ndim == 1
        assert new_dof_values.shape == (U.shape[0],)

        index = (Ellipsis,) + (np.newaxis,)*shape_range
        #U -= np.outer(new_dof_values, new_vec)
        U -= new_dof_values[index] * new_vec[np.newaxis,...]

    collateral_basis = np.array(collateral_basis)
    #interpolation_dofs_tuple = np.array(interpolation_dofs_tuple)
    interpolation_matrix = []
    for i in range(len(interpolation_dofs_tuple)):
        index = (slice(None),) + interpolation_dofs_tuple[i]
        #interpolation_matrix = collateral_basis[:, interpolation_dofs].T
        interpolation_matrix.append(collateral_basis[index].T)
    interpolation_matrix = np.array(interpolation_matrix)
    triangularity_errors = np.abs(interpolation_matrix - np.tril(interpolation_matrix))
    for d in range(1, len(interpolation_matrix)+1):
        triangularity_errs.append(np.max(triangularity_errors[:d, :d]))

    data = {'errors': max_errs, 'triangularity_errors': triangularity_errs}

    return interpolation_dofs, interpolation_dofs_tuple, collateral_basis, data





def interpolate_function_test(function, mus, xs):
    assert isinstance(function, FunctionInterface)
    #assert function.shape_range == tuple()

    assert isinstance(xs, np.ndarray)

    if function.shape_range == tuple():
        evaluations = np.array([function(xs, mu) for mu in mus])

        dofs, basis, data = ei_greedy_function(evaluations, target_error=1.0e-10, max_interpolation_dofs=25)

        f = EmpiricalInterpolatedFunction(function, dofs, basis, xs, triangular=True)

        return f

    else:
        evaluations = np.array([function(xs, mu) for mu in mus])
        dofs, dofs_tuple, basis, data = ei_greedy_function_COPY(evaluations, target_error=1.0e-10, max_interpolation_dofs=25)

        f = EmpiricalInterpolatedFunction(function, dofs_tuple, basis, xs, triangular=True)
        return f



def interpolate_function(function, samples, grid, points="center", target_error=None, max_interpolation_dofs=None):
    assert isinstance(function, FunctionInterface)
    #assert function.shape_range == tuple()

    assert isinstance(points, str)
    assert points in ["center", "quadrature"]

    samples = tuple(samples)

    #assert all([isinstance(mu, Parameter) for mu in samples])
    assert isinstance(grid, AffineGridWithOrthogonalCentersInterface)

    points = grid.centers(0) if points == "center" else grid.quadrature_points(0, order=2)
    points = points.ravel()

    evaluations = np.array([function(points, mu) for mu in samples])

    if function.shape_range == tuple():
        dofs, basis, data = ei_greedy_function(evaluations, target_error=target_error, max_interpolation_dofs=max_interpolation_dofs)
        f = EmpiricalInterpolatedFunction(function, dofs, basis, points, triangular=True)
    else:
        dofs, dofs_tuple, basis, data = ei_greedy_function_COPY(evaluations, target_error=target_error, max_interpolation_dofs=max_interpolation_dofs)
        f = EmpiricalInterpolatedFunction(function, dofs_tuple, basis, points, triangular=True)
    return f

    """
    mu_1 = argmax max ||f(x,mu)||
             mu    x

    x_1 = argmax ||f(x,mu_1)||
             x

    q_1 = f(., mu_1) / f(x_1,mu_1)

    """
    mu_1_index = np.argmax(np.max(np.abs(evaluations), axis=(1,2,3)))
    x_1_index = np.argmax(np.max(np.abs(evaluations[mu_1_index,...]), axis=(1,2)))
    q_1 = np.einsum('ij,xjk->xik', np.linalg.inv(evaluations[mu_1_index,x_1_index,...], evaluations[mu_1_index,...]))





