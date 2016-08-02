from pymor.functions.interfaces import FunctionInterface

from domaintransformation.functions.ei import EmpiricalInterpolatedFunction

from domaintransformation.functions.ei import EmpiricalInterpolatedProjectionFunction,\
    EmpiricalInterpolatedProjectionFunctional

import numpy as np


def ei_greedy_function(U, target_error=None, max_interpolation_dofs=None):
    assert isinstance(U, np.ndarray)
    assert len(U.shape) == 2  # mu x X

    interpolation_dofs = np.zeros((0,), dtype=np.int32)

    collateral_basis = []
    max_errs = []
    triangularity_errs = []
    _max_err_indeices = []
    _new_dof_indices = []
    _new_dof_values = []

    ERR = U

    while True:
        # Calculate maximum norm for x
        errs = np.max(np.abs(ERR), axis=1)

        assert len(errs.shape) == 1
        assert errs.shape == (ERR.shape[0],)

        max_err_ind = np.argmax(errs) # mu_m
        max_err = errs[max_err_ind]

        print("Basis size: {}".format(len(interpolation_dofs)))
        print("Max error: {}".format(max_err))

        if max_interpolation_dofs is not None and len(interpolation_dofs) >= max_interpolation_dofs:
            print("Maximum number interpolation dofs reached. Aborting...")
            break
        if target_error is not None and max_err <= target_error:
            print("Target Error reached. Aborting...")
            break

        #EIM
        new_vec = U[max_err_ind, ...].copy()
        assert len(new_vec.shape) == len(U.shape) - 1 # todo rather use ndim here
        assert new_vec.shape == U.shape[1:]

        new_dof_index = np.argmax(np.abs(new_vec))

        if new_dof_index in interpolation_dofs:
            print("DOF taken twice. Aborting...")
            break

        new_dof_value = new_vec[new_dof_index]  # x_m

        if new_dof_value == 0.:
            print("Error Zero")
            break

        new_vec *= 1 / new_dof_value  # q_m

        interpolation_dofs = np.hstack((interpolation_dofs, new_dof_index))
        collateral_basis.append(new_vec)
        max_errs.append(max_err)

        _max_err_indeices.append(max_err_ind)
        _new_dof_indices.append(new_dof_index)
        _new_dof_values.append(new_dof_value)

        # update U and ERR
        new_dof_values = U[:, new_dof_index]
        # f - I_f
        U -= np.outer(new_dof_values, new_vec)

    collateral_basis = np.array(collateral_basis)

    interpolation_matrix = collateral_basis[:, interpolation_dofs].T
    triangularity_errors = np.abs(interpolation_matrix - np.tril(interpolation_matrix))
    for d in range(1, len(interpolation_matrix)+1):
        triangularity_errs.append(np.max(triangularity_errors[:d, :d]))

    data = {'errors': max_errs, 'triangularity_errors': triangularity_errs, 'max_err_indices': _max_err_indeices,
            'new_dof_indices': _new_dof_indices, 'new_dof_values': _new_dof_values}

    return interpolation_dofs, collateral_basis, data


def mcei_greedy_function(U, target_error=None, max_interpolation_dofs=None):
    assert isinstance(U, np.ndarray)

    shape_range = U.ndim-1
    assert shape_range > 0

    interpolation_dofs = np.zeros((0,), dtype=np.int32)
    interpolation_dofs_tuple = []
    collateral_basis = []
    max_errs = []
    triangularity_errs = []
    _max_err_indices = []
    _new_dof_indices = []
    _new_dof_values = []

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

        print("Basis size: {}".format(len(interpolation_dofs)))
        print("Max error: {}".format(max_err))

        if max_interpolation_dofs is not None and len(interpolation_dofs) >= max_interpolation_dofs:
            print("Maximum number interpolation dofs reached. Aborting...")
            break
        if target_error is not None and max_err <= target_error:
            print("Target Error reached. Aborting...")
            break

        # EIM
        new_vec = U[max_err_ind,...].copy()
        assert new_vec.ndim == U.ndim - 1
        assert new_vec.shape == U.shape[1:]

        new_dof_index = np.argmax(np.abs(new_vec))

        # new_dof_index is a flattened index
        new_dof_index_tuple = np.unravel_index(new_dof_index, new_vec.shape)

        if new_dof_index in interpolation_dofs:
            print("DOF taken twice. Aborting...")
            break

        new_dof_value = new_vec[new_dof_index_tuple]  # x_m

        if new_dof_value == 0.:
            print("Error Zero. Aborting...")
            break

        new_vec *= 1 / new_dof_value  # q_m

        interpolation_dofs = np.hstack((interpolation_dofs, new_dof_index))
        interpolation_dofs_tuple.append(new_dof_index_tuple)

        collateral_basis.append(new_vec)
        max_errs.append(max_err)

        _max_err_indices.append(max_err_ind)
        _new_dof_indices.append(new_dof_index_tuple)
        _new_dof_values.append(new_dof_value)

        # update U and ERR
        index = (slice(None),) + new_dof_index_tuple
        assert len(index) == U.ndim

        new_dof_values = U[index]
        assert new_dof_values.ndim == 1
        assert new_dof_values.shape == (U.shape[0],)

        index = (Ellipsis,) + (np.newaxis,)*shape_range
        U -= new_dof_values[index] * new_vec[np.newaxis, ...]

    collateral_basis = np.array(collateral_basis)

    interpolation_matrix = []
    for i in range(len(interpolation_dofs_tuple)):
        index = (slice(None),) + interpolation_dofs_tuple[i]
        interpolation_matrix.append(collateral_basis[index].T)
    interpolation_matrix = np.array(interpolation_matrix)
    triangularity_errors = np.abs(interpolation_matrix - np.tril(interpolation_matrix))
    for d in range(1, len(interpolation_matrix)+1):
        triangularity_errs.append(np.max(triangularity_errors[:d, :d]))

    data = {'errors': max_errs, 'triangularity_errors': triangularity_errs, 'max_err_indices': _max_err_indices,
            'new_dof_indices': _new_dof_indices, 'new_dof_values': _new_dof_values}

    return interpolation_dofs, interpolation_dofs_tuple, collateral_basis, data


def split_ei_function(function):
    assert isinstance(function, EmpiricalInterpolatedFunction)
    num_functions = len(function.interpolation_dofs)

    print("Split function into {} functions/functionals".format(num_functions))

    f = function.function
    dofs = function.interpolation_dofs
    col_basis = function.collateral_basis
    xs = function.xs
    tri = function.triangular

    res = {'functions': [EmpiricalInterpolatedProjectionFunction(f, dofs, col_basis, xs, tri, i) for i in
                         range(num_functions)],
           'functionals': [EmpiricalInterpolatedProjectionFunctional(f, dofs, col_basis, xs, tri, i) for i in
                           range(num_functions)]}

    return res


def interpolate_function(function, mus, xs=None, evaluations=None, target_error=None, max_interpolation_dofs=None, mode='discrete'):
    import time
    start = time.time()
    assert isinstance(function, FunctionInterface)
    assert mode in ['discrete']

    mus = tuple(mus)

    if evaluations is not None:
        assert evaluations.shape == (len(mus), xs.shape[0]) + function.shape_range

    if evaluations is None:
        assert isinstance(xs, np.ndarray)

        print("Evaluate function for {} parameters".format(len(mus)))
        evaluations = []
        for i, mu in enumerate(mus):
            print("Evaluate function for parameter {}/{}".format(i+1, len(mus)))
            evaluations.append(function(xs, mu))
        evaluations = np.array(evaluations)

    if function.shape_range == tuple():
        dofs, basis, data = ei_greedy_function(evaluations, target_error=target_error,
                                               max_interpolation_dofs=max_interpolation_dofs)

        if mode == 'discrete':
            f = EmpiricalInterpolatedFunction(function, dofs, basis, xs, triangular=False)

        stop = time.time()
        print("Empirical Interpolation took {} seconds".format(stop-start))

        return f, data
    else:
        dofs, dofs_tuple, basis, data = mcei_greedy_function(evaluations, target_error=target_error,
                                                             max_interpolation_dofs=max_interpolation_dofs)

        assert len(data['new_dof_indices']) == len(data['max_err_indices']) == len(data['new_dof_values'])

        max_err_indices = data['max_err_indices']
        assert (all(index < len(mus) for index in max_err_indices))

        new_dof_indices = data['new_dof_indices']
        assert (all(index[0] < xs.shape[0]) for index in new_dof_indices)

        new_dof_values = data['new_dof_values']
        assert (all(isinstance(value, float)) for value in new_dof_values)

        if mode == 'discrete':
            f = EmpiricalInterpolatedFunction(function, dofs_tuple, basis, xs, triangular=False)

        stop = time.time()
        print("Empirical Interpolation took {} seconds".format(stop-start))
        return f, data
