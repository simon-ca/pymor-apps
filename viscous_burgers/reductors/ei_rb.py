# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import GenericRBReconstructor


def reduce_ei_rb(discretization, operator_names, data, operator_product=None, vector_product=None, disable_caching=True, extends=None):
    '''Combines EI-Interpolation with Generic reduced basis reductor.

    Reduces a discretization by applying `operators.project_operator` to
    each of its `operators`.

    Parameters
    ----------
    discretization
        The discretization which is to be reduced.
    data
        dict with the reduced basis, interpolation DOFs and collateral basis.
    product
        Scalar product for the projection. (See
        `operators.constructions.ProjectedOperator`)
    disable_caching
        If `True`, caching of the solutions of the reduced discretization
        is disabled.

    Returns
    -------
    rd
        The reduced discretization.
    rc
        The reconstructor providing a `reconstruct(U)` method which reconstructs
        high-dimensional solutions from solutions U of the reduced discretization.
    '''
    
    assert extends is None or len(extends) == 3
    
    operators = [discretization.operators[operator_name] for operator_name in operator_names]
    RB = data['RB']
    interpolation_dofs = data['dofs']
    collateral_basis = data['CB']
    
    ei_operators = {name: EmpiricalInterpolatedOperator(operator, interpolation_dofs, collateral_basis, triangular=True)
                    for name, operator in zip(operator_names, operators)}
    operators_dict = discretization.operators.copy()
    operators_dict.update(ei_operators)
    ei_discretization = discretization.with_(operators=operators_dict, name='{}_ei'.format(discretization.name))

    if RB is None:
        RB = discretization.type_solution.empty(discretization.dim_solution)
        
    projected_operators = {k: op.projected(source_basis=RB, range_basis=RB, product=operator_product) if op else None
                           for k, op in ei_discretization.operators.iteritems()}
    projected_functionals = {k: f.projected(source_basis=RB, range_basis=None, product=operator_product) if f else None
                             for k, f in ei_discretization.functionals.iteritems()}
    projected_vector_operators = {k: (op.projected(source_basis=None, range_basis=RB, product=vector_product) if op
                                      else None)
                                  for k, op in ei_discretization.vector_operators.iteritems()}

    if ei_discretization.products is not None:
        projected_products = {k: p.projected(source_basis=RB, range_basis=RB)
                              for k, p in ei_discretization.products.iteritems()}
    else:
        projected_products = None

    cache_region = None if disable_caching else ei_discretization.caching

    rd = ei_discretization.with_(operators=projected_operators, functionals=projected_functionals,
                              vector_operators=projected_vector_operators,
                              products=projected_products, visualizer=None, estimator=None,
                              cache_region=cache_region, name=discretization.name + '_reduced')
    rd.disable_logging()
    rc = GenericRBReconstructor(RB)

    return rd, rc, ei_discretization
