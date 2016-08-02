from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.interfaces import ImmutableInterface
from pymor.domaindescriptions.basic import RectDomain
from pymor.functions.basic import ConstantFunction


class EllipticRHSDecompositionProblem(ImmutableInterface):
    """
    Adds affinely decomposed right hand sides
    """

    def __init__(self, domain=RectDomain(), rhs_functions=None,
                 rhs_functionals=None,
                 diffusion_functions=None,
                 diffusion_functionals=None,
                 advection_functions=None,
                 advection_functionals=None,
                 reaction_functions=None,
                 reaction_functionals=None,
                 dirichlet_data=None, neumann_data=None, robin_data=None,
                 parameter_space=None, name=None):

        assert rhs_functions is None or isinstance(rhs_functions, (tuple, list))
        assert diffusion_functions is None or isinstance(diffusion_functions, (tuple, list))
        assert advection_functions is None or isinstance(advection_functions, (tuple, list))
        assert reaction_functions is None or isinstance(reaction_functions, (tuple, list))

        assert rhs_functionals is None and rhs_functions is None \
            or rhs_functionals is None and len(rhs_functions) == 1 \
            or len(rhs_functionals) == len(rhs_functions)
        assert diffusion_functionals is None and diffusion_functions is None \
            or diffusion_functionals is None and len(diffusion_functions) == 1 \
            or len(diffusion_functionals) == len(diffusion_functions)
        assert advection_functionals is None and advection_functions is None \
            or advection_functionals is None and len(advection_functions) == 1 \
            or len(advection_functionals) == len(advection_functions)
        assert reaction_functionals is None and reaction_functions is None \
            or reaction_functionals is None and len(reaction_functions) == 1 \
            or len(reaction_functionals) == len(reaction_functions)

        # for backward compatibility:
        if (diffusion_functions is None and advection_functions is None and reaction_functions is None):
            diffusion_functions = (ConstantFunction(dim_domain=2),)

        # dim_domain:
        if diffusion_functions is not None:
            dim_domain = diffusion_functions[0].dim_domain

        for r in rhs_functions:
            assert r.dim_domain == dim_domain
        if diffusion_functions is not None:
            for f in diffusion_functions:
                assert f.dim_domain == dim_domain
        if advection_functions is not None:
            for f in advection_functions:
                assert f.dim_domain == dim_domain
        if reaction_functions is not None:
            for f in reaction_functions:
                assert f.dim_domain == dim_domain

        assert dirichlet_data is None or dirichlet_data.dim_domain == dim_domain
        assert neumann_data is None or neumann_data.dim_domain == dim_domain
        assert robin_data is None or (isinstance(robin_data, tuple) and len(robin_data) == 2)
        assert robin_data is None or np.all([f.dim_domain == dim_domain for f in robin_data])
        self.domain = domain
        self.rhs_functions = rhs_functions
        self.rhs_functionals = rhs_functionals
        self.diffusion_functions = diffusion_functions
        self.diffusion_functionals = diffusion_functionals
        self.advection_functions = advection_functions
        self.advection_functionals = advection_functionals
        self.reaction_functions = reaction_functions
        self.reaction_functionals = reaction_functionals
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.robin_data = robin_data
        self.parameter_space = parameter_space
        self.name = name