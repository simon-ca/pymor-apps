from __future__ import absolute_import, division, print_function

from pymor.analyticalproblems.elliptic import EllipticProblem

from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.functions.basic import ConstantFunction

#from pymor.discretizers.elliptic import discretize_elliptic_cg
from domaintransformation.discretizers.elliptic import discretize_elliptic_cg

from domaintransformation.functions.transformation import DomainTransformationFunction, DiffusionTransformation, JacobianDeterminantTransformation
from domaintransformation.functions.basic import ProductFunction

from domaintransformation.grids.transformation import DomainTransformationTriaGrid

def discretize_elliptic_cg_ei(analytical_problem, transformation, diameter=None, domain_discretizer=None, grid=None,
                              boundary_info=None, options=None):
    """Discretizes an |EllipticProblem| using finite elements.
    Parameters
    ----------
    analytical_problem
        The |EllipticProblem| to discretize.
    transformation
        A |DomainTransformationFunction| which is applied to the domain of the analytical prolem
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
    options
        A list in which each entry specifies how to decompose the functions
    Returns
    -------
    list of
        discretization
            The |Discretization| that has been generated.
        data
            Dictionary with the following entries:
                :grid:           The generated |Grid|.
                :boundary_info:  The generated |BoundaryInfo|.
                :interpolation:  The interpolation method which were used for the functions
    """

    ALLOWED_OPTIONS = [None, "eim", "mceim", "eoi"]

    assert isinstance(analytical_problem, EllipticProblem)

    # only basic problems can be transformed for now
    # todo update pymor and handle advections and reactions functions
    assert analytical_problem.diffusion_functionals is None
    assert len(analytical_problem.diffusion_functions) == 1
    assert analytical_problem.diffusion_functions[0] is None\
           or isinstance(analytical_problem.diffusion_functions[0], ConstantFunction)
    # todo handle non constant right hand sides
    assert analytical_problem.rhs is None or isinstance(analytical_problem.rhs, ConstantFunction)
    # todo data_functions
    assert analytical_problem.dirichlet_data is None
    assert analytical_problem.neumann_data is None
    # todo update pymor and handle robin_data
    #assert analytical_problem.robin_data is None
    # todo handle parameter_space
    assert analytical_problem.parameter_space is None

    assert isinstance(transformation, DomainTransformationFunction)

    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    assert options is None or isinstance(options, list)
    options = options or [None]
    assert all(opt in ALLOWED_OPTIONS for opt in options)

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    grid_trafo = DomainTransformationTriaGrid(grid, transformation)
    discretizations = []
    for opt in options:
        if opt is None:
            print("Don't interpolate the functions")
            diffusion_functions = [DiffusionTransformation(transformation)]
            diffusion_functionals = None
            jac_det =  JacobianDeterminantTransformation(transformation)
            rhs = ProductFunction(analytical_problem.rhs, jac_det)
        elif opt == "eim":
            print("Interpolating each component of the functions separately using EIM")
            raise NotImplementedError
        elif opt == "mceim":
            print("Interpolating multi-dimensional functions using multi component EIM")
            raise NotImplementedError
        elif opt == "eoi":
            print("Interpolationg the operators using empirical operator interpolation")
            raise NotImplementedError

        domain = analytical_problem.domain
        parameter_space = transformation.parameter_space
        name = analytical_problem.name

        problem = EllipticProblem(domain, rhs=rhs, diffusion_functions=diffusion_functions,
                                  diffusion_functionals=diffusion_functionals, parameter_space=parameter_space,
                                  name=name)

        discretization, data = discretize_elliptic_cg(problem, grid=grid_trafo, boundary_info=boundary_info)
        data["interpolation"] = opt
        data['transformation_grid'] = grid_trafo
        discretizations.append((discretization, data))

    return discretizations



