from __future__ import print_function

from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import GenericFunction, ConstantFunction
from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.grids.tria import TriaGrid
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.core.pickle import dump

import numpy as np

"""
def test():
    domain = RectDomain()
    rhs = GenericFunction(lambda X: -5.0 * np.pi * np.sin(np.pi * X[...,0]) * np.cos(2.0 * np.pi * X[..., 1]), 2)
    dirichlet = GenericFunction(lambda X: np.sin(np.pi * X[...,0]) * np.cos(2.0 * np.pi * X[..., 1]), 2)

    for n in [32, 128]:
        grid_name = '{1}(({0},{0}))'.format(n, 'TriaGrid')
        print('Solving on {0}'.format(grid_name))

        print('Setup problem ...')
        problem = EllipticProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet)

        print('Discretize ...')
        grid, bi = discretize_domain_default(problem.domain, diameter=1. / n, grid_type=TriaGrid)
        discretizer = discretize_elliptic_cg
        discretization, _ = discretizer(analytical_problem=problem, grid=grid, boundary_info=bi)

        print('Solve ...')
        U = discretization.solve()

        print('Plot ...')
        discretization.visualize(U, title=grid_name)

        print('')
"""

import sys

from pymor.reductors.basic import reduce_generic_rb, reduce_to_subbasis

from pymor.analyticalproblems.elliptic import EllipticProblem

from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.grids.tria import TriaGrid
from pymor.parameters.base import Parameter

from domaintransformation.discretizers.transformation import discretize_elliptic_cg_ei
from domaintransformation.functions.ffd import FreeFormDeformation, DIRECTION_X, DIRECTION_Y, DIRECTION_BOTH,\
    DIRECTION_NONE, CONSTRAINT_SAME, CONSTRAINT_INVERSE, CONSTRAINT_SAME_MIRROR, CONSTRAINT_INVERSE_MIRROR

import numpy as np
from matplotlib import pyplot as plt

def setup_analytical_problem():
    domain = RectDomain(domain=([0, 0], [1, 1]))
    rhs = GenericFunction(lambda X: -5.0 * np.pi * np.sin(np.pi * X[...,0]) * np.cos(2 * np.pi * X[..., 1]), 2)
    dirichlet = GenericFunction(lambda X: np.sin(np.pi * X[...,0]) * np.cos(2 * np.pi * X[..., 1]), 2)
    #dirichlet = ConstantFunction(value=np.array(0.0), dim_domain=2)

    problem = EllipticProblem(domain=domain, rhs=rhs, dirichlet_data=dirichlet, name="elliptic")
    return problem


def setup_ffd(problem, num_control_points, active, constraints, shift_min=None, shift_max=None, ranges=None):
    """
    def parameter_mask(num_control_points, active=[]):
        assert isinstance(num_control_points, tuple)
        assert len(num_control_points) == 2

        K, L = num_control_points

        assert isinstance(active, list)
        assert len(active) <= K*L*2

        assert all(isinstance(a, tuple) for a in active)
        assert all(len(a) == 3 for a in active)
        assert all(isinstance(a, int) and isinstance(b, int) and isinstance(c, int) for a, b, c in active)
        assert all(x >= 0 and x < K and y >= 0 and y < L and z in [0, 1, 2, 3] for x, y, z in active)

        mask = np.zeros((K, L), dtype=np.int)
        for x, y, z in active:
            if z == 0:
                mask[x, y] += DIRECTION_X
            elif z == 1:
                mask[x, y] += DIRECTION_Y
            elif z == 2:
                mask[x, y] += DIRECTION_BOTH
            else:
                raise NotImplementedError

        return mask
    """

    domain = problem.domain
    # mask = parameter_mask(num_control_points, active)
    K, L = num_control_points

    transformation = FreeFormDeformation(domain, K, L, active, constraints, "ffd", shift_min, shift_max, ranges)

    return transformation


def setup_grid(Nx, Ny):
    num_intervals = (Nx, Ny)
    grid = TriaGrid(num_intervals, domain=([0, 0], [1, 1]))
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


def setup_demo_2(options=[None], samples=4, target_error=None, Nx=None, Ny=None):

    num_control_points = (3, 3)
    active = [(2, 1, DIRECTION_X),
              (1, 2, DIRECTION_Y),
              (2, 2, DIRECTION_BOTH),
              ]
    constraints = [None,
                   (0, CONSTRAINT_SAME, CONSTRAINT_SAME_MIRROR),
                   None]


    shift_min = -0.15
    shift_max = 0.15
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


def setup_demo_1(options=[None], samples=4, target_error=None):
    Nx = 50
    Ny = 50

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


def compare_solutions(disc_1, disc_2, mus):
    errs_l2 = []
    errs_h1 = []

    for i, mu in enumerate(mus):
        print("Solving for parameter {}/{}".format(i+1, len(mus)))
        U_FEM = disc_1.solve(mu)
        U_EIM = disc_2.solve(mu)
        ERR = U_FEM - U_EIM

        l2_err = disc_1.l2_norm(ERR)
        h1_err = disc_1.h1_norm(ERR)

        errs_l2.append(l2_err)
        errs_h1.append(h1_err)

    errs_l2 = np.array(errs_l2)
    errs_h1 = np.array(errs_h1)

    return {'l2': errs_l2, 'h1': errs_h1}

def plot_solution(discretization, mus, separate=False, rescale=False):
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

def plot_solution_matplotlib(discretization, grid, mus, cmap='afmhot', transform=True, rescale=False):
    Us = [discretization.solve(mu) for mu in mus]
    Us = [U.data[0] for U in Us]
    triangles = grid.subentities(0, 2)

    x = grid.centers(2)[..., 0]
    y = grid.centers(2)[..., 1]

    bbox = grid.bounding_box(tuple(mus))
    x_lim = bbox[..., 0]
    y_lim = bbox[..., 1]

    if rescale:
        vmin = min([U.min() for U in Us])
        vmax = max([U.max() for U in Us])
    else:
        vmin = None
        vmax = None

    for i in range(len(mus)):
        U = Us[i]
        mu = mus[i]

        if transform:
            c = grid.centers(2, mu=mu)
            x = c[..., 0]
            y = c[..., 1]
        else:
            c = grid.centers(2)
            x = c[..., 0]
            y = c[..., 1]

        title = "Solution for mu = {} on {}x{}".format(mu, Nx, Ny)
        plt.figure(title)
        plt.tripcolor(x, y, triangles, U, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    z = 0

def reduce_basis(discretization):
    from pymor.algorithms.basisextension import trivial_basis_extension, gram_schmidt_basis_extension
    from pymor.algorithms.greedy import greedy
    from pymor.reductors.stationary import reduce_stationary_coercive
    from pymor.reductors.linear import reduce_stationary_affine_linear
    from functools import partial

    pool = None

    estimator_norm = 'h1'
    SNAPSHOTS = 6
    reductor = 'residual_basis'
    extension_alg = 'h1_gram_schmidt'
    without_estimator = False
    RBSIZE = 40
    print('RB generation ...')

    error_product = discretization.h1_product if estimator_norm == 'h1' else None
    #coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', discretization.parameter_type)
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
                         discretization.parameter_space.sample_uniformly(SNAPSHOTS),
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


def plot_configuration(trafo, mus):
    x = np.linspace(0, 1, 100)
    zero = np.zeros_like(x)
    one = np.ones_like(x)

    left = np.array((zero, x)).T
    right = np.array((one, x)).T
    bottom = np.array((x, zero)).T
    top = np.array((x, one)).T

    for mu in mus:
        title = "Configuration for {}".format(mu)

        l = trafo.evaluate(left, mu=mu)
        r = trafo.evaluate(right, mu=mu)
        b = trafo.evaluate(bottom, mu=mu)
        t = trafo.evaluate(top, mu=mu)

        P_mu = trafo.P_0 + trafo._assemble_parameter(mu)
        P_mu = P_mu.reshape((-1, 2))

        domain = [[0, 0], [1, 1]]
        domain = np.array(domain)

        bbox = trafo.bounding_box(domain=domain, mu=mu)

        margin = [-0.1, 0.1]
        margin = np.array(margin)

        xlim = bbox[..., 0] + margin
        ylim = bbox[..., 1] + margin

        plt.figure(title)

        plt.plot(l[..., 0], l[..., 1], color='b')
        plt.plot(r[..., 0], r[..., 1], color='b')
        plt.plot(b[..., 0], b[..., 1], color='b')
        plt.plot(t[..., 0], t[..., 1], color='b')

        plt.scatter(P_mu[..., 0], P_mu[..., 1], color='r')

        plt.ylim(ylim)
        plt.xlim(xlim)

    z = 0


if __name__ == "__main__":
    Nx = 50
    Ny = 50

    options_reference = [None]
    target_error_reference = None

    options_eim = ['mceim']
    target_errors_eim = [1.0e-2]

    samples = 8

    discretizations_reference = setup_demo_2(options_reference, samples, target_error_reference, Nx=Nx, Ny=Ny)
    assert isinstance(discretizations_reference, list)
    assert len(discretizations_reference) == 1
    assert isinstance(discretizations_reference[0], tuple)
    assert len(discretizations_reference[0]) == 2

    discretization_reference, data_reference = discretizations_reference[0]
    grid = reference = data_reference['grid']
    trafo = data_reference['transformation_grid'].transformation

    mu = [0.0, 0.0, 0.0]
    mu = Parameter({'ffd': mu})
    mus = discretization_reference.parameter_space.sample_randomly(2)

    mus = [mu] + list(mus)

    #plot_configuration(trafo, mus)
    #plot_solution(discretization_reference, mus, rescale=True)
    #plot_solution_matplotlib(discretization=discretization_reference,
    #                         grid=grid,
    #                         mus=mus,
    #                         rescale=True)

    discretizations_eim = []
    for target_error in target_errors_eim:
        discretizations = setup_demo_2(options_eim, samples, target_error, Nx=Nx, Ny=Ny)
        discretizations_eim.append((discretizations, target_error))

    disc_eim = [disc_[0][0] for disc_ in discretizations_eim if disc_[0][0][1]['interpolation'] in ["mceim"]]
    disc_eim = [disc_[0][0] for disc_ in discretizations_eim]
    disc_eim = sorted(disc_eim, key=lambda d: d[1]['error_tolerance'], reverse=True)

    assert isinstance(disc_eim, list)
    assert len(disc_eim) == len(target_errors_eim)
    assert all(isinstance(disc, tuple) for disc in disc_eim)
    assert all(len(disc) == 2 for disc in disc_eim)

    #discretization_eim = [disc_[0] for disc_ in disc_eim]

    test_mus = list(discretization_reference.parameter_space.sample_randomly(100))
    test_Us = [discretization_reference.solve(mu=mu) for mu in test_mus]

    results = []
    for d_mceim, d_data in disc_eim:
        print("Error tolerance: ", d_data['error_tolerance'])
        #print("Number of functions:", len(d_mceim.operator.operators))
        #res = evaluate_eim(disc_reference, d_mceim, test_mus, test_Us)
        d_red, rc, extensions = reduce_basis(d_mceim)
        res = evaluate_rb(d_mceim, d_red, rc, extensions, test_mus, test_Us)

        results.append({'d_eim': d_mceim, 'd_data': d_data, 'd_red': d_red, 'rc': rc,
                        'extensions': extensions, 'res': res})

    print("Interpolated on {} parameters".format(samples**3))
    print("Evaluated on {} parameters".format(len(test_mus)))
    print()


    f = file("/home/simon/Dokumente/results.data", 'wb')
    dump(results, f)
    f.close()

    z = 0

