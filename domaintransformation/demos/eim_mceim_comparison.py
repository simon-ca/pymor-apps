from __future__ import print_function

import time
from pymor.analyticalproblems.elliptic import EllipticProblem

from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.grids.tria import TriaGrid
from pymor.parameters.base import Parameter

from domaintransformation.discretizers.transformation import discretize_elliptic_cg_ei
from domaintransformation.functions.basic import ProjectionFunction, MergeFunction
from domaintransformation.functions.ffd import FreeFormDeformation, DIRECTION_X, DIRECTION_Y, DIRECTION_BOTH,\
    DIRECTION_NONE, CONSTRAINT_SAME, CONSTRAINT_INVERSE

from domaintransformation.functions.transformation import DiffusionTransformation
from domaintransformation.algorithms.ei import interpolate_function

import numpy as np
from matplotlib import pyplot as plt

def setup_analytical_problem():
    problem = EllipticProblem(name="elliptic")
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
    Nx = 80
    Ny = 80

    num_control_points = (3, 3)
    active = [(0, 0, DIRECTION_BOTH),
              (1, 0, DIRECTION_Y),
              (2, 0, DIRECTION_BOTH),
              (0, 1, DIRECTION_X),
              (2, 1, DIRECTION_X),
              (0, 2, DIRECTION_BOTH),
              (1, 2, DIRECTION_Y),
              (2, 2, DIRECTION_BOTH),
              ]
    constraints = [None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None]

    shift_min = -0.2
    shift_max = 0.2
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

def plot_configuration(trafo, mus):
    center = np.array((0.5, 0.5))
    radius = 0.4

    l = np.linspace(0, 1, 100)

    x = np.linspace(center[0]-radius, center[0]+radius, 100)
    y = np.sqrt(radius**2 - (x-center[0])**2)
    y_top = y+center[1]
    y_bot = -y + center[1]

    bot = np.array((x, y_bot)).T
    top = np.array((x, y_top)).T

    boundary = np.vstack((bot, top[::-1]))

    for mu in mus:
        title = "Configuration for {}".format(mu)

        b = trafo.evaluate(boundary, mu=mu)

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
        plt.plot(b[..., 0], b[..., 1], color='b')
        plt.scatter(P_mu[..., 0], P_mu[..., 1], color='r')

        plt.ylim(ylim)
        plt.xlim(xlim)

    z = 0


def plot_basis_sizes(eims, error_tolerances):
    assert len(eims) == len(error_tolerances)

    basis_sizes_mceim = [len(eim['mceim'][1]['errors']) for eim in eims]
    bs_0_0 = [len(eim['0_0'][1]['errors']) for eim in eims]
    bs_0_1 = [len(eim['0_1'][1]['errors']) for eim in eims]
    bs_1_0 = [len(eim['1_0'][1]['errors']) for eim in eims]
    bs_1_1 = [len(eim['1_1'][1]['errors']) for eim in eims]
    basis_sizes_eim = [bs_0_0[i] + bs_0_1[i] + bs_1_0[i] + bs_1_1[i] for i in range(len(bs_0_0))]

    plt.figure("Basis sizes")
    plt.semilogx(error_tolerances, basis_sizes_mceim, label="MCEIM")
    plt.semilogx(error_tolerances, basis_sizes_eim, label="EIM")

    plt.legend()

    plt.xlabel("$\\varepsilon$", fontsize=16)
    plt.ylabel("$M$", fontsize=16)

    z = 0

def timings(d_ref, eims, x, mus):
    d_mceims = [eim['mceim'][0] for eim in eims]
    d_eims = [eim['eim'][0] for eim in eims]

    t = {'reference': np.zeros((len(mus),)),
         'mceim': np.zeros((len(eims), len(mus))),
         'eim': np.zeros((len(eims), len(mus)))}

    for i_mu, mu in enumerate(mus):
        print("Solve uninterpolated function for parameter {}/{}".format(i_mu+1, len(mus)))
        tic = time.time()
        d_ref.evaluate(x, mu)
        toc = time.time()
        t['reference'][i_mu] = toc - tic

    for i_tol, d_mceim in enumerate(d_mceims):
        d_mceim.evaluate(x, mus[0])
        for i_mu, mu in enumerate(mus):
            print("Solve MCEIM-interpolated function for tolerance {}/{} and parameter {}/{}".format(i_tol+1, len(d_mceims),                                                                                         i_mu+1, len(mus)))
            tic = time.time()
            d_mceim.evaluate(x, mu)
            toc = time.time()
            t['mceim'][i_tol, i_mu] = toc - tic

    for i_tol, d_eim in enumerate(d_eims):
        d_eim.evaluate(x, mus[0])
        for i_mu, mu in enumerate(mus):
            print("Solve EIM-interpolated function for tolerance {}/{} and parameter {}/{}".format(i_tol+1, len(d_mceims),
                                                                                                   i_mu+1, len(mus)))
            tic = time.time()
            d_eim.evaluate(x, mu)
            toc = time.time()
            t['eim'][i_tol, i_mu] = toc - tic

    np.savetxt("/home/simon/Dokumente/timings_reference.txt", t['reference'])
    np.savetxt("/home/simon/Dokumente/timings_mceim.txt", t['mceim'])
    np.savetxt("/home/simon/Dokumente/timings_eim.txt", t['eim'])

    return t




if __name__ == "__main__":
    options_reference = [None]
    target_error_reference = None

    samples = 10

    discretizations_reference = setup_demo_1(options_reference, samples, target_error_reference)
    assert isinstance(discretizations_reference, list)
    assert len(discretizations_reference) == 1
    assert isinstance(discretizations_reference[0], tuple)
    assert len(discretizations_reference[0]) == 2

    discretization_reference, data_reference = discretizations_reference[0]
    grid = data_reference['grid']
    trafo = data_reference['transformation_grid'].transformation

    mu = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mu = Parameter({'ffd': mu})

    mus = discretization_reference.parameter_space.sample_randomly(2)
    mus = [mu] + list(mus)

    plot_configuration(trafo, mus)

    d = DiffusionTransformation(trafo)

    d_0_0 = ProjectionFunction(d, (0, 0))
    d_0_1 = ProjectionFunction(d, (0, 1))
    d_1_0 = ProjectionFunction(d, (1, 0))
    d_1_1 = ProjectionFunction(d, (1, 1))

    grid = TriaGrid((50, 50))
    x = grid.centers(0)

    mus_eim = list(discretization_reference.parameter_space.sample_randomly(500))
    mus_test = list(discretization_reference.parameter_space.sample_randomly(100))

    evaluations = []
    for i, mu in enumerate(mus_eim):
        print("Evaluating function for parameter {}/{}".format(i+1, len(mus_eim)))
        evaluations.append(d(x, mu))
    evaluations = np.array(evaluations)

    error_tolerances = [1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5]
    #error_tolerances = [1.0E-1]
    max_basis_size = None

    eims = []
    for error_tolerance in error_tolerances:
        e = evaluations.copy()

        d_mceim, d_data = interpolate_function(d, mus=mus_eim, xs=x, evaluations=e, target_error=error_tolerance,
                                               max_interpolation_dofs=max_basis_size)
        del e
        e_0_0 = evaluations[..., 0, 0].copy()
        d_ei_0_0, d_ei_0_0_data = interpolate_function(d_0_0, mus=mus_eim, xs=x, evaluations=e_0_0,
                                                       target_error=error_tolerance,
                                                       max_interpolation_dofs=max_basis_size)
        del e_0_0
        e_0_1 = evaluations[..., 0, 1].copy()
        d_ei_0_1, d_ei_0_1_data = interpolate_function(d_0_1, mus=mus_eim, xs=x, evaluations=e_0_1,
                                                       target_error=error_tolerance,
                                                       max_interpolation_dofs=max_basis_size)
        del e_0_1
        e_1_0 = evaluations[..., 1, 0].copy()
        d_ei_1_0, d_ei_1_0_data = interpolate_function(d_1_0, mus=mus_eim, xs=x, evaluations=e_1_0,
                                                       target_error=error_tolerance,
                                                       max_interpolation_dofs=max_basis_size)
        del e_1_0
        e_1_1 = evaluations[..., 1, 1].copy()
        d_ei_1_1, d_ei_1_1_data = interpolate_function(d_1_1, mus=mus_eim, xs=x, evaluations=e_1_1,
                                                       target_error=error_tolerance,
                                                       max_interpolation_dofs=max_basis_size)
        del e_1_1
        eims.append({"mceim": (d_mceim, d_data),
                     "0_0": (d_ei_0_0, d_ei_0_0_data),
                     "0_1": (d_ei_0_1, d_ei_0_1_data),
                     "1_0": (d_ei_1_0, d_ei_1_0_data),
                     "1_1": (d_ei_1_1, d_ei_1_1_data),
                     "eim": (d_ei_1_1, d_ei_1_1_data)})

    for i, eim in enumerate(eims):
        print("Error tolerance: ", error_tolerances[i])
        print("mceim", len(eim['mceim'][1]['errors']))
        print("0_0", len(eim['0_0'][1]['errors']))
        print("0_1", len(eim['0_1'][1]['errors']))
        print("1_0", len(eim['1_0'][1]['errors']))
        print("1_1", len(eim['1_1'][1]['errors']))
        print()

    #plot_basis_sizes(eims, error_tolerances)
    t = timings(d, eims, x, mus_test)

    z = 0
