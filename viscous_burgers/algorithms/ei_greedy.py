# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import time

import numpy as np
from scipy.linalg import solve_triangular

from pymor.core import getLogger
from pymor.core.exceptions import ExtensionError
from pymor.algorithms.basisextension import trivial_basis_extension
from pymor.algorithms.ei import EvaluationProvider
from pymor.reductors import reduce_generic_rb
from reductors.ei_rb import reduce_ei_rb
from pymor.la import VectorArrayInterface


def ei_rb_greedy(discretization, operator_names, samples, error_norm=None, target_error=None,
                 product=None, max_extensions=None, rb_initial_data=None, ei_initial_data=None,
                 use_estimator=True, extension_algorithm=trivial_basis_extension, cache_region='memory'):
    '''PODEI Greedy extension algorithm.

    Parameters
    ----------
    discretization
        The discretization to reduce.
    reductor
        Reductor for reducing the given discretization. This has to be a
        function of the form `reduce(discretization, data)` where data is
        the detailed data required by the reductor. If your reductor takes
        more arguments, use functools.partial.
    samples
        The set of parameter samples on which to perform the greedy search.
        Currently this set is fixed for the whole process.
    initial_data
        This is fed into reductor.reduce() for the initial projection.
        Typically this will be the reduced basis with which the algorithm
        starts.
    use_estimator
        If True, use reduced_discretization.estimate() to estimate the errors
        on the sample set. Otherwise a detailed simulation is used to calculate
        the error.
    error_norm
        If use_estimator == Flase, use this function to calculate the norm of
        the error. [Default l2_norm]
    extension_algorithm
        The extension algorithm to use to extend the current reduced basis with
        the maximum error snapshot.
    target_error
        If not None, stop the search if the maximum error on the sample set
        drops below this value.
    max_extensions
        If not None, stop algorithm after `max_extensions` extension steps.

    Returns
    -------
    Dict with the following fields:
        'data'
            The reduced basis. (More generally the data which needs to be
            fed into reduced_discretization.reduce().
        'reduced_discretization'
            The last reduced discretization which has been computed.
        'reconstructor'
            Reconstructor for `reduced_discretization`.
        'max_err'
            Last estimated maximum error on the sample set.
        'max_err_mu'
            The parameter that corresponds to `max_err`.
        'max_errs'
            Sequence of maximum errors during the greedy run.
        'max_errs_mu'
            The parameters corresponding to `max_err`.
    '''

    def interpolate(U, ind=None):
        coefficients = solve_triangular(interpolation_matrix, U.components(interpolation_dofs, ind=ind).T,
                                        lower=True, unit_diagonal=True).T
        # coefficients = np.linalg.solve(interpolation_matrix, U.components(interpolation_dofs, ind=ind).T).T
        return collateral_basis.lincomb(coefficients)

    tic = time.time()

    logger = getLogger('pymor.algorithms.ei_greedy.ei_greedy')
    samples = tuple(samples)
    logger.info('Started PODEI greedy search on {} samples'.format(len(samples)))
    data = rb_initial_data

    #ei init
    operators = [discretization.operators[operator_name] for operator_name in operator_names]
    evaluations = EvaluationProvider(discretization, operators, samples, cache_region=cache_region)

    assert isinstance(evaluations, VectorArrayInterface) or all(isinstance(ev, VectorArrayInterface) for ev in evaluations)
    if isinstance(evaluations, VectorArrayInterface):
        evaluations = (evaluations,)

    if ei_initial_data is None:
        interpolation_dofs = np.zeros((0,), dtype=np.int32)
        interpolation_matrix = np.zeros((0,0))
        collateral_basis = type(next(iter(evaluations))).empty(dim=next(iter(evaluations)).dim)
    else:
        interpolation_dofs = ei_initial_data['dofs']
        collateral_basis = ei_initial_data['basis']
        interpolation_matrix = collateral_basis.components(interpolation_dofs).T

    extensions = 0
    discard_count = 0
    crb_discard = False
    error_already_calculated = False
    Ns = [len(data)]
    Ms = [len(interpolation_dofs)]
    max_errs = []
    max_err_mus = []

    while True:
        logger.info('Reducing ...')
        
        if len(interpolation_dofs) > 0:
            rd, rc, _ = reduce_ei_rb(discretization, operator_names, data={'RB': data, 'dofs': interpolation_dofs, 'CB': collateral_basis})
        else:
            rd, rc = reduce_generic_rb(discretization, data)

        logger.info('Estimating errors ...')
        if use_estimator:
            logger.info('Error Estimator usage not yet implemented')
            break
        elif not error_already_calculated:
            max_err = -1
            for mu in samples:
                errs = error_norm(discretization.solve(mu) - rc.reconstruct(rd.solve(mu)))
                cur_max_err = np.max(errs)
                if cur_max_err > max_err:
                    max_err = cur_max_err
                    max_err_mu = mu
                    max_err_t = np.argmax(errs)      
            
        max_errs.append(max_err)
        max_err_mus.append(max_err_mu)
        logger.info('Maximum error after {} extensions: {} (mu = {}, timestep t_k: k={})'.format(extensions, max_err, max_err_mu, max_err_t))
        
        if target_error is not None and max_err <= target_error:
            logger.info('Target error reached! Stopping extension loop.')
            logger.info('Reached maximal error on snapshots of {} <= {}'.format(max_err, target_error))
            break

        # compute new interpolation dof and collateral basis vector
        ev = evaluations.data(samples.index(max_err_mu))
        if len(interpolation_dofs) > 0:
            ev_interpolated = interpolate(ev)
            ERR = ev - ev_interpolated
        else:
            ERR = ev
        new_vec = ERR.copy(ind=np.argmax(error_norm(ERR)))
     
        if new_vec.amax()[1] > 1.0e-2:      
            new_dof = new_vec.amax()[0][0]
            if new_dof in interpolation_dofs:
                logger.info('DOF {} selected twice for interplation! Stopping extension loop.'.format(new_dof))
                break
            new_vec *= 1 / new_vec.components([new_dof])[0, 0]
            interpolation_dofs = np.hstack((interpolation_dofs, new_dof))
            collateral_basis.append(new_vec, remove_from_other=True)
            interpolation_matrix = collateral_basis.components(interpolation_dofs).T
            crb_discard = False
        else:
            logger.warn('Maximum DOF is {}, skipping collateral basis extension ...'.format(new_vec.amax()[1]))
            crb_discard = True

        triangularity_error = np.max(np.abs(interpolation_matrix - np.tril(interpolation_matrix)))
        logger.info('Interpolation matrix is not lower triangular with maximum error of {}'
                    .format(triangularity_error))
        
        logger.info('Extending with snapshot for mu = {}'.format(max_err_mu))
        U = discretization.solve(max_err_mu)
        try:
            data, extension_data = extension_algorithm(data, U)
        except ExtensionError:
            logger.info('Extension failed. Stopping now.')
            break
        if not 'hierarchic' in extension_data:
            logger.warn('Extension algorithm does not report if extension was hierarchic. Assuming it was\'nt ..')
        
        rd, rc, _ = reduce_ei_rb(discretization, operator_names, data={'RB': data, 'dofs': interpolation_dofs, 'CB': collateral_basis})
        if use_estimator:
            logger.info('Error Estimator usage not yet implemented')
            break
        else:
            max_err = -1
            for mu in samples:
                errs = error_norm(discretization.solve(mu) - rc.reconstruct(rd.solve(mu)))
                cur_max_err = np.max(errs)
                if cur_max_err > max_err:
                    max_err = cur_max_err
                    max_err_mu = mu
                    max_err_t = np.argmax(errs)
        if max_errs[len(max_errs)-1] <= max_err and not crb_discard:
            logger.info('Error Increases. Discard last RB extension')
            data.remove(ind=[len(data)-1])
            discard_count += 1
            error_already_calculated = False
        else:
            extensions += 1
            Ns.append(len(data))
            Ms.append(len(interpolation_dofs))
            discard_count = 0
            error_already_calculated = True
                    
        #Version 2
#         if max_errs[len(max_errs)-1] <= max_err:
#             if discard_count < 1:
#                 logger.info('Error Increases. Discard last RB extension')
#                 data.remove(ind=[len(data)-1])
#                 discard_count = 1
#                 error_already_calculated = False
#             else:
#                 logger.info('Error still increases. Throwing away CB but keeping RB.')
#                 extensions += 1
#                 collateral_basis.remove([len(interpolation_dofs)-1])
#                 interpolation_dofs = interpolation_dofs[:-1]
#                 interpolation_matrix = interpolation_matrix[:-1,:-1]
#                 Ns.append(len(data))
#                 Ms.append(len(interpolation_dofs))
#                 discard_count = 0
#                 error_already_calculated = False
#         else:
#             extensions += 1
#             Ns.append(len(data))
#             Ms.append(len(interpolation_dofs))
#             discard_count = 0
#             error_already_calculated = True
            
        #Version 1
#         if max_errs[len(max_errs)-1] <= max_err:
#             if discard_count < 1:
#                 logger.info('Error Increases. Discard last RB extension')
#                 data.remove(ind=[len(data)-1])
#                 discard_count = 1
#                 error_already_calculated = False
#             else:
#                 logger.info('Error still increases. Keeping RB.')
#                 extensions += 1
#                 Ns.append(len(data))
#                 Ms.append(len(interpolation_dofs))
#                 discard_count = 0
#                 error_already_calculated = True
#         else:
#             extensions += 1
#             Ns.append(len(data))
#             Ms.append(len(interpolation_dofs))
#             discard_count = 0
#             error_already_calculated = True

        logger.info('N={}, M={}'.format(len(data), len(interpolation_dofs)))
        logger.info('')
        
        if max_extensions is not None and extensions >= max_extensions:
            logger.info('Maximal number of {} extensions reached.'.format(max_extensions))
            break
    
    logger.info('Reducing once more ...')
    rd, rc, ei_discretization = reduce_ei_rb(discretization, operator_names, data={'RB':data, 'dofs': interpolation_dofs, 'CB': collateral_basis})

    ei_data = ({'dofs': interpolation_dofs, 'basis': collateral_basis})
    N_M_correlation = [Ns,Ms]

    tictoc = time.time() - tic
    logger.info('PODEI Greedy search took {} seconds'.format(tictoc))
    return {'ei_discretization': ei_discretization, 'ei_data': ei_data, 
            'data': data, 'reduced_discretization': rd, 'reconstructor': rc, 'max_err': max_err,
            'max_err_mu': max_err_mu, 'max_errs': max_errs, 'max_err_mus': max_err_mus, 'extensions': extensions, 'N_M_correlation': N_M_correlation,
            'time': tictoc}

