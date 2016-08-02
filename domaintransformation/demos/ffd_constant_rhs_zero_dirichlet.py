from __future__ import print_function
import sys

from pymor.reductors.basic import reduce_to_subbasis

from pymor.analyticalproblems.elliptic import EllipticProblem

from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.grids.tria import TriaGrid
from pymor.parameters.base import Parameter

from domaintransformation.discretizers.transformation import discretize_elliptic_cg_ei
from domaintransformation.functions.ffd import FreeFormDeformation, DIRECTION_X, DIRECTION_Y, DIRECTION_BOTH,\
    DIRECTION_NONE, CONSTRAINT_SAME, CONSTRAINT_INVERSE

import numpy as np
from matplotlib import pyplot as plt


def setup_analytical_problem():
    problem = EllipticProblem(name="elliptic")
    return problem


def setup_ffd(problem, num_control_points, active, constraints, shift_min=None, shift_max=None, ranges=None):
    domain = problem.domain
    # mask = parameter_mask(num_control_points, active)
    K, L = num_control_points

    transformation = FreeFormDeformation(domain, K, L, active, constraints, "ffd", shift_min, shift_max, ranges)

    return transformation


def setup_grid(Nx, Ny):
    num_intervals = (Nx, Ny)
    grid = TriaGrid(num_intervals)
    bi = AllDirichletBoundaryInfo(grid)

    return grid, bi


def setup_discretizations(problem, transformation, grid, boundary_info, samples, target_error, options, mode):
    discretizations = discretize_elliptic_cg_ei(problem, transformation, grid=grid, boundary_info=boundary_info,
                                                samples=samples, target_error=target_error, options=options, mode=mode)

    return discretizations


def setup_main(grid_options, ffd_options, ranges_options, ei_options, discretization_options):
    assert isinstance(grid_options, dict)
    assert len(grid_options) == 2
    assert 'Nx' in grid_options.keys()
    assert 'Ny' in grid_options.keys()

    assert isinstance(ffd_options, dict)
    assert len(ffd_options) == 3
    assert 'num_control_points' in ffd_options.keys()
    assert 'active' in ffd_options.keys()
    assert 'constraints' in ffd_options.keys()

    assert isinstance(ranges_options, dict)
    assert len(ranges_options) == 3
    assert 'shift_min' in ranges_options.keys()
    assert 'shift_max' in ranges_options.keys()
    assert 'ranges' in ranges_options.keys()

    assert isinstance(ei_options, dict)
    assert len(ei_options) == 3
    assert 'samples' in ei_options.keys()
    assert 'target_error' in ei_options.keys()
    assert 'max_basis_size' in ei_options.keys()

    assert isinstance(discretization_options, dict)
    assert len(discretization_options) == 2
    assert 'options' in discretization_options.keys()
    assert 'mode' in discretization_options.keys()

    Nx = grid_options['Nx']
    Ny = grid_options['Ny']
    assert isinstance(Nx, int)
    assert isinstance(Ny, int)
    assert Nx > 0
    assert Ny > 0

    num_control_points = ffd_options['num_control_points']
    assert isinstance(num_control_points, tuple)
    assert len(num_control_points) == 2

    K, L = num_control_points

    assert isinstance(K, int)
    assert isinstance(L, int)
    assert K > 0
    assert L > 0

    active = ffd_options['active']
    constraints = ffd_options['constraints']

    shift_min = ranges_options['shift_min']
    shift_max = ranges_options['shift_max']
    ranges = ranges_options['ranges']
    assert ranges is None and shift_min is not None and shift_max is not None or\
        ranges is not None and shift_min is None and shift_max is None

    samples = ei_options['samples']
    target_error = ei_options['target_error']
    max_basis_size = ei_options['max_basis_size']

    options = discretization_options['options']
    mode = discretization_options['mode']

    problem = setup_analytical_problem()
    transformation = setup_ffd(problem=problem, num_control_points=num_control_points, active=active,
                               constraints=constraints, shift_min=shift_min, shift_max=shift_max, ranges=ranges)
    grid, boundary_info = setup_grid(Nx, Ny)

    discretizations = setup_discretizations(problem, transformation, grid, boundary_info, samples, target_error,
                                            options, mode)

    return discretizations


def setup_demo_1(options=[None], samples=4, target_error=None):
    Nx = 40
    Ny = 40

    num_control_points = (3, 2)
    active = [(0, 0, DIRECTION_Y),
              (1, 0, DIRECTION_Y),
              (2, 0, DIRECTION_Y),
              (0, 1, DIRECTION_Y),
              (1, 1, DIRECTION_Y),
              (2, 1, DIRECTION_Y)]
    constraints = [None,
                   None,
                   (0, CONSTRAINT_SAME, CONSTRAINT_SAME),
                   (0, CONSTRAINT_SAME, CONSTRAINT_INVERSE),
                   (1, CONSTRAINT_SAME, CONSTRAINT_INVERSE),
                   (0, CONSTRAINT_SAME, CONSTRAINT_INVERSE)]

    shift_min = -0.4
    shift_max = 0.4
    ranges = None

    samples = samples
    max_basis_size = None

    options = options
    mode = 'discrete'

    grid_options = {'Nx': Nx, 'Ny': Ny}
    ffd_options = {'num_control_points': num_control_points, 'active': active, 'constraints': constraints}
    ranges_options = {'shift_min': shift_min, 'shift_max': shift_max, 'ranges': ranges}
    ei_options = {'samples': samples, 'target_error': target_error, 'max_basis_size': max_basis_size}
    discretization_options = {'options': options, 'mode': mode}

    discretizations = setup_main(grid_options, ffd_options, ranges_options, ei_options, discretization_options)

    return discretizations


def plot_solutions(discretization, mus, separate=False, rescale=False):
    mus = list(mus)
    Us = [discretization.solve(mu=mu) for mu in mus]
    Us = tuple(Us)
    mus = tuple(mus)
    legend = ["Soution for mu = {}".format(mu) for mu in mus]
    legend = tuple(legend)
    discretization.visualize(U=Us,
                             mu=mus,
                             legend=legend, separate_colorbars=separate,
                             rescale_colorbars=rescale,
                             title="Example solutions for different parameters")


def reduce_basis(discretization):
    from pymor.algorithms.basisextension import trivial_basis_extension, gram_schmidt_basis_extension
    from pymor.algorithms.greedy import greedy
    from pymor.reductors.stationary import reduce_stationary_coercive
    from pymor.reductors.linear import reduce_stationary_affine_linear
    from functools import partial

    pool = None

    estimator_norm = 'h1'

    reductor = 'residual_basis'
    extension_alg = 'h1_gram_schmidt'
    without_estimator = False

    print('RB generation ...')

    error_product = discretization.h1_product if estimator_norm == 'h1' else None

    coercivity_estimator = None
    reductors = {'residual_basis': partial(reduce_stationary_coercive, error_product=error_product,
                                           coercivity_estimator=coercivity_estimator),
                 'traditional': partial(reduce_stationary_affine_linear, error_product=error_product,
                                        coercivity_estimator=coercivity_estimator)}
    reductor = reductors[reductor]
    extension_algorithms = {'trivial': trivial_basis_extension,
                            'gram_schmidt': gram_schmidt_basis_extension,
                            'h1_gram_schmidt': partial(gram_schmidt_basis_extension, product=discretization.h1_product)}
    extension_algorithm = extension_algorithms[extension_alg]

    greedy_data = greedy(discretization, reductor,
                         discretization.parameter_space.sample_uniformly(RBSNAPSHOTS),
                         use_estimator=not without_estimator, error_norm=discretization.h1_norm,
                         extension_algorithm=extension_algorithm, max_extensions=RBSIZE,
                         pool=pool)

    rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']
    extensions = greedy_data['extensions']

    return rb_discretization, reconstructor, extensions


def evaluate_rb(discretization, discretization_rb, reconstructor, extensions, test_mus, test_Us):

    Us = test_Us

    max_l2 = max([discretization.l2_norm(U)[0] for U in Us])
    max_h1 = max([discretization.h1_norm(U)[0] for U in Us])

    l2_abs_errs = np.zeros(shape=(extensions, len(test_mus)))
    h1_abs_errs = np.zeros(shape=(extensions, len(test_mus)))

    for i in range(1, extensions+1):
        rd, rc, _ = reduce_to_subbasis(discretization_rb, i, reconstructor)
        #rd = rd.with_(operator=rd.operator.projected_to_subbasis(dim_collateral=M))

        for j, mu in enumerate(test_mus): # i-> j
            print('.', end='')
            sys.stdout.flush()
            u = rd.solve(mu)
            URB = rc.reconstruct(u)
            U = Us[j] # i -> j
            #U = discretization.solve(mu)
            l2_abs = discretization.l2_norm(U - URB)[0]
            l2_abs = np.inf if not np.isfinite(l2_abs) else l2_abs
            h1_abs = discretization.h1_norm(U - URB)[0]
            h1_abs = np.inf if not np.isfinite(h1_abs) else h1_abs

            l2_abs_errs[i-1, j] = l2_abs
            h1_abs_errs[i-1, j] = h1_abs

        print()

    return {'l2': l2_abs_errs, 'h1': h1_abs_errs, 'max_l2': max_l2, 'max_h1': max_h1}


def evaluate_eim(discretization, discretization_eim, test_mus, test_Us):

    Us = test_Us

    max_l2 = max([discretization.l2_norm(U)[0] for U in Us])
    max_h1 = max([discretization.h1_norm(U)[0] for U in Us])

    l2_abs_errs = np.zeros(shape=(len(test_mus), 1))
    h1_abs_errs = np.zeros(shape=(len(test_mus), 1))

    for j, mu in enumerate(test_mus): # i-> j
        print("Solving for parameter {}/{}".format(j+1, len(test_mus)))
        sys.stdout.flush()
        UEIM = discretization_eim.solve(mu)
        U = Us[j] # i -> j

        l2_abs = discretization.l2_norm(U - UEIM)[0]
        l2_abs = np.inf if not np.isfinite(l2_abs) else l2_abs
        h1_abs = discretization.h1_norm(U - UEIM)[0]
        h1_abs = np.inf if not np.isfinite(h1_abs) else h1_abs

        l2_abs_errs[j, 0] = l2_abs
        h1_abs_errs[j, 0] = h1_abs

        print()

    return {'l2': l2_abs_errs, 'h1': h1_abs_errs, 'max_l2': max_l2, 'max_h1': max_h1}


if __name__ == "__main__":
    options_reference = [None]
    target_error_reference = None

    options_eim = ['mceim_projection']
    target_errors_eim = [1.0e-1, 1.0e-3, 1.0e-5]
    target_errors_eim = [1.0e-1, 1.0e-2, 1.0e-4, 1.0e-5]
    target_errors_eim = target_errors_eim[:1]

    mu_samples = 4
    RBSNAPSHOTS = 10
    RBSIZE = 20

    discretizations_reference = (setup_demo_1(options_reference, mu_samples, target_error_reference), target_error_reference)

    disc_reference, _ = discretizations_reference[0][0]

    mus = [[0, 0], [0.2, 0.15], [-0.3, -0.2]]

    mus = [Parameter({'ffd': mu}) for mu in mus]

    plot_solutions(disc_reference, mus, separate=True, rescale=True)

    discretizations_eim = []
    for target_error in target_errors_eim:
        discretizations = setup_demo_1(options_eim, mu_samples, target_error)
        discretizations_eim.append((discretizations, target_error))

    disc_mceim = [disc_[0][0] for disc_ in discretizations_eim if disc_[0][0][1]['interpolation'] in ["mceim_projection"]]
    disc_mceim = [disc_[0][0] for disc_ in discretizations_eim]
    disc_mceim = sorted(disc_mceim, key=lambda d: d[1]['error_tolerance'], reverse=True)

    test_mus = list(disc_reference.parameter_space.sample_randomly(10))
    test_Us = [disc_reference.solve(mu=mu) for mu in test_mus]

    results = []
    for d_mceim, d_data in disc_mceim:
        print("Error tolerance: ", d_data['error_tolerance'])

        d_red, rc, extensions = reduce_basis(d_mceim)
        res = evaluate_rb(d_mceim, d_red, rc, extensions, test_mus, test_Us)

        results.append((res, d_data))

    print("Interpolated on {} parameters".format(mu_samples ** 2))
    print("Evaluated on {} parameters".format(len(test_mus)))
    print()

    plt.figure("Errors in H1")
    for res, d_data in results:
        x = np.arange(1, res['h1'].shape[0]+1)
        y = res['h1'].max(axis=1)
        plt.plot(x, y, label="$H^1$")

    plt.legend()



