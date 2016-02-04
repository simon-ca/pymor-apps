from pymor.functions.interfaces import FunctionInterface
from pymor.functions.basic import ConstantFunction
from domaintransformation.functions.ei import EmpiricalInterpolatedFunction
from domaintransformation.functions.ei import EIFunction

from pymor.parameters.base import Parameter

from pymor.grids.interfaces import AffineGridWithOrthogonalCentersInterface

from domaintransformation.functions.ei import FixedParameterFunction

import numpy as np

def ei_greedy_function(U, target_error=None, max_interpolation_dofs=None):
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


def mcei_greedy_function(U, target_error=None, max_interpolation_dofs=None):
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


def interpolate_function(function, mus, xs, target_error=None, max_interpolation_dofs=None):
    assert isinstance(function, FunctionInterface)
    #assert function.shape_range == tuple()

    assert isinstance(xs, np.ndarray)

    mus = tuple(mus)

    evaluations = np.array([function(xs, mu) for mu in mus])

    if function.shape_range == tuple():
        dofs, basis, data = ei_greedy_function(evaluations, target_error=target_error,
                                               max_interpolation_dofs=max_interpolation_dofs)

        f = EmpiricalInterpolatedFunction(function, dofs, basis, xs, triangular=False)

        return f
    else:
        dofs, dofs_tuple, basis, data = mcei_greedy_function(evaluations, target_error=target_error,
                                                             max_interpolation_dofs=max_interpolation_dofs)

        f = EmpiricalInterpolatedFunction(function, dofs_tuple, basis, xs, triangular=False)
        return f


def interpolate_function_analytically(function, mus, xs, target_error=None, max_interpolation_dofs=None):
    assert isinstance(function, FunctionInterface)
    assert function.shape_range == tuple()

    mus = tuple(mus)
    assert all([isinstance(mu, Parameter) for mu in mus])
    assert isinstance(xs, np.ndarray)
    assert xs.ndim == function.dim_domain
    #assert xs.shape[1] == function.dim_domain

    interpolants_f = [ConstantFunction(value=0.0, dim_domain=function.dim_domain)]
    interpolants_xi = [FixedParameterFunction(ConstantFunction(value=0.0), mus[0])]
    qs = []
    ts = []

    while True:
        mu_m = None
        f_max = 0
        for mu in mus:
            for x in xs:
                f_eval = function(x, mu) - interpolants_f[-1](x, mu)
                assert isinstance(f_eval, float)
                f_eval = np.abs(f_eval)
                if f_eval > f_max:
                    mu_m = mu
                    f_max = f_eval

        assert mu_m is not None
        assert isinstance(mu_m, Parameter)

        xi_m = FixedParameterFunction(function, mu_m)

        t_m = None
        xi_max = 0
        for x in xs:
            xi_eval = xi_m(x) - interpolants_xi[-1](x)
            xi_eval = np.abs(xi_eval)
            if xi_eval > xi_max:
                t_m = x
                xi_max = xi_eval
        assert t_m is not None
        assert isinstance(t_m, float)

        if t_m in ts:
            print("Using same DOF twice, Aborting...")
            break

        assert t_m is not None
        denominator = xi_m - interpolants_xi[-1]
        numerator = xi_m(t_m) - interpolants_xi[-1](t_m)
        q_m = denominator*(1.0/numerator)

        qs.append(q_m)
        ts.append(t_m)

        interpolants_f.append(EIFunction(function, qs, ts))
        interpolants_xi.append(EIFunction(xi_m, qs, ts))







