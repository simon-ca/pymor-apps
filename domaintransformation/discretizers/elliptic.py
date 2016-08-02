from __future__ import absolute_import, division, print_function


from domaintransformation.analyticalproblems.rhs_decomposition import EllipticRHSDecompositionProblem
from pymor.discretizations.basic import StationaryDiscretization
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.grids.referenceelements import line, triangle, square
from domaintransformation.gui.qt import DomainTransformationPatchVisualizer
from pymor.operators import cg
from domaintransformation.operators import cg as cg_rhs_decomp
from pymor.operators.constructions import LincombOperator


def discretize_elliptic_cg(analytical_problem, diameter=None, domain_discretizer=None,
                           grid=None, boundary_info=None):
    """Discretizes an |EllipticProblem| using finite elements.

    Parameters
    ----------
    analytical_problem
        The |EllipticProblem| to discretize.
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

    Returns
    -------
    discretization
        The |Discretization| that has been generated.
    data
        Dictionary with the following entries:

            :grid:           The generated |Grid|.
            :boundary_info:  The generated |BoundaryInfo|.
    """

    assert isinstance(analytical_problem, EllipticRHSDecompositionProblem)
    #assert grid is not None
    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    assert grid.reference_element in (line, triangle, square)

    if grid.reference_element is square:
        Operator = cg.DiffusionOperatorQ1
        Functional = cg_rhs_decomp.L2ProductFunctionalQ1
    else:
        Operator = cg.DiffusionOperatorP1
        Functional = cg_rhs_decomp.L2ProductFunctionalP1

    p = analytical_problem

    if p.diffusion_functionals is not None:
        L0 = Operator(grid, boundary_info, diffusion_constant=0, name='diffusion_boundary_part')

        Li = [Operator(grid, boundary_info, diffusion_function=df, dirichlet_clear_diag=True,
                       name='diffusion_{}'.format(i))
              for i, df in enumerate(p.diffusion_functions)]

        L = LincombOperator(operators=[L0] + Li, coefficients=[1.] + list(p.diffusion_functionals),
                            name='diffusion')
    else:
        assert len(p.diffusion_functions) == 1
        L = Operator(grid, boundary_info, diffusion_function=p.diffusion_functions[0],
                     name='diffusion')

    if p.rhs_functionals is not None:
        #boundary treatment
        F0 = Functional(grid, p.rhs_functions[0], boundary_info, dirichlet_data=p.dirichlet_data, neumann_data=p.neumann_data,
                        name='rhs_boundary', clear_dirichlet=False, clear_non_dirichlet=True)
        Fi = [Functional(grid, rhs, boundary_info, dirichlet_data=p.dirichlet_data, neumann_data=p.neumann_data,
                         name='rhs_{}'.format(i), clear_dirichlet=True, clear_non_dirichlet=False) for i, rhs in enumerate(p.rhs_functions)]
        F = LincombOperator(operators=[F0] + Fi, coefficients=[1.] + list(p.rhs_functionals),
                            name='rhs')
    else:
        F = Functional(grid, p.rhs_functions[0], boundary_info, dirichlet_data=p.dirichlet_data, neumann_data=p.neumann_data)

    if grid.reference_element in (triangle, square):
        visualizer = DomainTransformationPatchVisualizer(grid=grid, bounding_box=grid.bounding_box(), codim=2)
    else:
        visualizer = None

    Prod = cg.L2ProductQ1 if grid.reference_element is square else cg.L2ProductP1
    empty_bi = EmptyBoundaryInfo(grid)
    l2_product = Prod(grid, empty_bi)
    l2_0_product = Prod(grid, boundary_info, dirichlet_clear_columns=True)
    h1_semi_product = Operator(grid, empty_bi)
    h1_0_semi_product = Operator(grid, boundary_info, dirichlet_clear_columns=True)
    products = {'h1': l2_product + h1_semi_product,
                'h1_semi': h1_semi_product,
                'l2': l2_product,
                'h1_0': l2_0_product + h1_0_semi_product,
                'h1_0_semi': h1_0_semi_product,
                'l2_0': l2_0_product}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_CG'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}

