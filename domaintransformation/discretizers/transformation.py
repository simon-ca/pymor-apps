from __future__ import absolute_import, division, print_function

from pymor.analyticalproblems.elliptic import EllipticProblem
from domaintransformation.analyticalproblems.rhs_decomposition import EllipticRHSDecompositionProblem

from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.functions.basic import ConstantFunction

from domaintransformation.discretizers.elliptic import discretize_elliptic_cg

from domaintransformation.functions.transformation import DomainTransformationFunction, DiffusionTransformation, JacobianDeterminantTransformation
from domaintransformation.functions.basic import ProductFunction, ProjectionFunction, MergeFunction, WideningFunction

from domaintransformation.grids.transformation import DomainTransformationTriaGrid

from domaintransformation.algorithms.ei import interpolate_function, split_ei_function


def discretize_elliptic_cg_ei(analytical_problem, transformation, diameter=None, domain_discretizer=None, grid=None,
                              boundary_info=None, samples=None, target_error=None, options=None, mode='discrete'):
    """Discretizes an |EllipticProblem| using finite elements.
    Parameters
    ----------
    analytical_problem
        The |EllipticProblem| to discretize.
    transformation
        A |DomainTransformationFunction| which is applied to the domain of the analytical problem
    diameter
        If not `None`, `diameter` is passed to the `domain_discretizer`.
    domain_discretizer
        Discretizer to be used for discretizing the analytical domain. This has
        to be a function `domain_discretizer(domain_description, diameter, ...)`.
        If further arguments should be passed to the discretizer, use
        :func:`functools.partial`. If `None`, |discretize_domain_default| is used.
    grid
        Instead of using a domain discretizer, the |Grid| can also be passed directly
        using this parameter.
    boundary_info
        A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
        Must be provided if `grid` is specified.
    samples
        Number of parameter snapshots for the empirical interpolation
    target_error
        Maximum error for the interpolation
    options
        A list in which each entry specifies how to decompose the functions
    mode
        How to interpolate the Function. Only 'discrete' supported atm
    Returns
    -------
    list of
        discretization
            The |Discretization| that has been generated.
        data
            Dictionary with the following entries:
                :grid:           The generated |Grid|.
                :boundary_info:  The generated |BoundaryInfo|.
                :interpolation:  The interpolation method which was used for the functions
    """

    MCEIM_OPTIONS = ["mceim", "mceim_projection"]
    EIM_OPTIONS = ["eim", "eim_projection"]
    ALLOWED_OPTIONS = [None] + MCEIM_OPTIONS + EIM_OPTIONS + ["eoi"]
    ALLOWED_MODES = ['discrete', 'half-analytical']

    assert isinstance(analytical_problem, EllipticProblem)

    # only basic problems can be transformed for now
    assert analytical_problem.diffusion_functionals is None
    assert len(analytical_problem.diffusion_functions) == 1
    assert analytical_problem.diffusion_functions[0] is None \
           or isinstance(analytical_problem.diffusion_functions[0], ConstantFunction)
    assert analytical_problem.neumann_data is None
    #assert analytical_problem.robin_data is None
    assert analytical_problem.parameter_space is None

    assert isinstance(transformation, DomainTransformationFunction)

    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    assert options is None or isinstance(options, list)
    options = options or [None]
    assert all(opt in ALLOWED_OPTIONS for opt in options)
    assert mode in ALLOWED_MODES

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    grid_trafo = DomainTransformationTriaGrid(grid, transformation)

    # transform diffusion
    if analytical_problem.diffusion_functions is None:
        diffusion_function = DiffusionTransformation(transformation)
        diffusion_functions = [diffusion_function]
    else:
        diff_function = analytical_problem.diffusion_functions[0]
        diffusion_function = ProductFunction(DiffusionTransformation(transformation), diff_function)
        diffusion_functions = [diffusion_function]

    # transform RHS
    jac_det = JacobianDeterminantTransformation(transformation)
    rhs = ProductFunction(analytical_problem.rhs, jac_det)
    dirichlet_data = analytical_problem.dirichlet_data

    discretizations = []

    # interpolate functions only once
    if any([opt in MCEIM_OPTIONS for opt in options]):
        # mceim
        print("Interpolating multi-dimensional functions using MCEIM")
        assert len(diffusion_functions) == 1

        target_error = target_error or 1.0e-5
        max_interpolation_dofs = None
        ei_samples = samples or 2

        func = diffusion_functions[0]
        mus = tuple(transformation.parameter_space.sample_uniformly(ei_samples))
        print("Number of Parameters: {}".format(len(mus)))

        xs = grid.centers(0)
        assert xs.shape[-1] == func.dim_domain

        func_ei, func_ei_data = interpolate_function(func, mus, xs, evaluations=None, target_error=target_error, max_interpolation_dofs=max_interpolation_dofs, mode=mode)

    if any([opt in EIM_OPTIONS for opt in options]):
        # eim
        print("Interpolating each component of the functions separately using EIM")
        assert len(diffusion_functions) == 1

        target_error = target_error or 1.0e-5
        max_interpolation_dofs = None
        ei_samples = samples or 2

        func = diffusion_functions[0]

        # only function in R^(2,2)
        func_0_0 = ProjectionFunction(func, (0, 0))
        func_0_1 = ProjectionFunction(func, (0, 1))
        func_1_0 = ProjectionFunction(func, (1, 0))
        func_1_1 = ProjectionFunction(func, (1, 1))
        mus = tuple(transformation.parameter_space.sample_uniformly(ei_samples))
        print("Number of Parameters: {}".format(len(mus)))

        xs = grid.centers(0)
        assert xs.shape[-1] == func.dim_domain

        func_ei_0_0, func_ei_0_0_data = interpolate_function(func_0_0, mus, xs, evaluations=None, target_error=target_error, max_interpolation_dofs=max_interpolation_dofs, mode=mode)
        func_ei_0_1, func_ei_0_1_data = interpolate_function(func_0_1, mus, xs, evaluations=None, target_error=target_error, max_interpolation_dofs=max_interpolation_dofs, mode=mode)
        func_ei_1_0, func_ei_1_0_data = interpolate_function(func_1_0, mus, xs, evaluations=None, target_error=target_error, max_interpolation_dofs=max_interpolation_dofs, mode=mode)
        func_ei_1_1, func_ei_1_1_data = interpolate_function(func_1_1, mus, xs, evaluations=None, target_error=target_error, max_interpolation_dofs=max_interpolation_dofs, mode=mode)

    if any([opt in EIM_OPTIONS or opt in MCEIM_OPTIONS for opt in options]):
        # rhs
        print("Interpolating rhs using EIM")
        assert len(diffusion_functions) == 1

        target_error = target_error or 1.0e-5
        max_interpolation_dofs = None
        ei_samples = samples or 2

        func = diffusion_functions[0]
        mus = tuple(transformation.parameter_space.sample_uniformly(ei_samples))
        print("Number of Parameters: {}".format(len(mus)))

        xs = grid.centers(0)
        assert xs.shape[-1] == func.dim_domain

        rhs_ei, rhs_ei_data = interpolate_function(rhs, mus, xs, evaluations=None, target_error=target_error, max_interpolation_dofs=max_interpolation_dofs, mode=mode)

    for opt in options:
        if opt is None:
            print("Don't interpolate the functions")
            diffusion_functions_ = diffusion_functions
            diffusion_functionals_ = None
            rhs_functions_ = [rhs]
            rhs_functionals_ = None

            assert len(diffusion_functions_) == 1
            assert diffusion_functionals_ is None
        elif opt == "mceim":
            diffusion_functions_ = [func_ei]
            diffusion_functionals_ = None

            rhs_functions_ = [rhs_ei]
            rhs_functionals_ = None

            print("Collateral Basis size diffusion: {}".format(len(func_ei.collateral_basis)))
            print("Collateral Basis size RHS: {}".format(len(rhs_ei.collateral_basis)))

            assert len(diffusion_functions_) == 1
            assert diffusion_functionals_ is None
        elif opt == "mceim_projection":

            func_ei_split = split_ei_function(func_ei)
            diffusion_functions_ = func_ei_split['functions']
            diffusion_functionals_ = func_ei_split['functionals']

            rhs_ei_split = split_ei_function(rhs_ei)
            rhs_functions_ = rhs_ei_split['functions']
            rhs_functionals_ = rhs_ei_split['functionals']

            print("Collateral Basis size diffusion: {}".format(len(func_ei.collateral_basis)))
            print("Collateral Basis size RHS: {}".format(len(rhs_ei.collateral_basis)))
        elif opt == "eim":
            func_ei = MergeFunction({(0, 0): func_ei_0_0, (0, 1): func_ei_0_1, (1, 0): func_ei_1_0, (1, 1): func_ei_1_1})
            diffusion_functions_ = [func_ei]
            diffusion_functionals_ = None

            rhs_functions_ = [rhs_ei]
            rhs_functionals_ = None

            print("Collateral Basis size diffusion (0, 0): {}".format(len(func_ei_0_0.collateral_basis)))
            print("Collateral Basis size diffusion (0, 1): {}".format(len(func_ei_0_1.collateral_basis)))
            print("Collateral Basis size diffusion (1, 0): {}".format(len(func_ei_1_0.collateral_basis)))
            print("Collateral Basis size diffusion (1, 1): {}".format(len(func_ei_1_1.collateral_basis)))
            print("Collateral Basis size RHS: {}".format(len(rhs_ei.collateral_basis)))

            assert len(diffusion_functions_) == 1
            assert diffusion_functionals_ is None
        elif opt == "eim_projection":
            func_ei_0_0_split = split_ei_function(func_ei_0_0)
            func_ei_0_1_split = split_ei_function(func_ei_0_1)
            func_ei_1_0_split = split_ei_function(func_ei_1_0)
            func_ei_1_1_split = split_ei_function(func_ei_1_1)

            sr = (2, 2)

            func_ei_0_0_wide = [WideningFunction(f, sr, (0, 0)) for f in func_ei_0_0_split['functions']]
            func_ei_0_1_wide = [WideningFunction(f, sr, (0, 1)) for f in func_ei_0_1_split['functions']]
            func_ei_1_0_wide = [WideningFunction(f, sr, (1, 0)) for f in func_ei_1_0_split['functions']]
            func_ei_1_1_wide = [WideningFunction(f, sr, (1, 1)) for f in func_ei_1_1_split['functions']]

            func_ei_wide = func_ei_0_0_wide + func_ei_0_1_wide + func_ei_1_0_wide + func_ei_1_1_wide
            functionals_wide = func_ei_0_0_split['functionals'] + func_ei_0_1_split['functionals'] +\
                               func_ei_1_0_split['functionals'] + func_ei_1_1_split['functionals']

            diffusion_functions_ = func_ei_wide
            diffusion_functionals_ = functionals_wide

            rhs_ei_split = split_ei_function(rhs_ei)
            rhs_functions_ = rhs_ei_split['functions']
            rhs_functionals_ = rhs_ei_split['functionals']

            print("Collateral Basis size diffusion (0, 0): {}".format(len(func_ei_0_0.collateral_basis)))
            print("Collateral Basis size diffusion (0, 1): {}".format(len(func_ei_0_1.collateral_basis)))
            print("Collateral Basis size diffusion (1, 0): {}".format(len(func_ei_1_0.collateral_basis)))
            print("Collateral Basis size diffusion (1, 1): {}".format(len(func_ei_1_1.collateral_basis)))
            print("Collateral Basis size RHS: {}".format(len(rhs_ei.collateral_basis)))

            assert len(diffusion_functions_) == len(diffusion_functionals_)

        domain = analytical_problem.domain
        parameter_space = transformation.parameter_space
        name = analytical_problem.name


        problem = EllipticRHSDecompositionProblem(domain, rhs_functions=rhs_functions_,
                                                  rhs_functionals=rhs_functionals_,
                                                  diffusion_functions=diffusion_functions_,
                                                  diffusion_functionals=diffusion_functionals_,
                                                  dirichlet_data=dirichlet_data,
                                                  parameter_space=parameter_space, name=name)


        if opt in MCEIM_OPTIONS:
            ind = func_ei_data['max_err_indices']
            mus_diffusion = [mus[i] for i in ind]

            ind = rhs_ei_data['max_err_indices']
            mus_rhs = [mus[i] for i in ind]
        elif opt in EIM_OPTIONS:
            ind_0_0 = func_ei_0_0_data['max_err_indices']
            ind_0_1 = func_ei_0_1_data['max_err_indices']
            ind_1_0 = func_ei_1_0_data['max_err_indices']
            ind_1_1 = func_ei_1_1_data['max_err_indices']

            mus_diffusion_0_0 = [mus[i] for i in ind_0_0]
            mus_diffusion_0_1 = [mus[i] for i in ind_0_1]
            mus_diffusion_1_0 = [mus[i] for i in ind_1_0]
            mus_diffusion_1_1 = [mus[i] for i in ind_1_1]

            mus_diffusion = {(0,0): mus_diffusion_0_0, (0,1): mus_diffusion_0_1,
                             (1,0): mus_diffusion_1_0, (1,1): mus_diffusion_1_1}

            ind = rhs_ei_data['max_err_indices']
            mus_rhs = [mus[i] for i in ind]
        else:
            mus_diffusion = None
            mus_rhs = None

        discretization, data = discretize_elliptic_cg(problem, grid=grid_trafo, boundary_info=boundary_info)
        data["interpolation"] = opt
        data['transformation_grid'] = grid_trafo
        data['reference_grid'] = grid
        data['error_tolerance'] = target_error
        data['mus'] = {'diffusion': mus_diffusion, 'rhs': mus_rhs}
        discretizations.append((discretization, data))

    return discretizations



