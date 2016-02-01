import numpy as np

from pymor.grids.tria import TriaGrid
from pymor.grids.unstructured import UnstructuredTriangleGrid

from pymor.discretizers.elliptic import discretize_elliptic_fv

from pymor.analyticalproblems.elliptic import EllipticProblem

from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo, BoundaryInfoFromIndicators

from pymor.parameters.base import Parameter
from pymor.parameters.spaces import CubicParameterSpace

from pymor.functions.basic import GenericFunction, ConstantFunction

from domaintransformation.functions.basic import DomainTransformationFunction
from domaintransformation.analyticalproblems.elliptic_transformation import AffineTransformationProblem
from domaintransformation.grids.domaintransformation import DomainTransformationTriaGrid

from domaintransformation.discretizers.elliptic import discretize_elliptic_cg

import matplotlib.pyplot as plt

class AffineTransformation(DomainTransformationFunction):
    dim_domain = 2
    shape_range = (2, 2)

    def __init__(self, parameter_type, min=None, max=None, ranges=None):
        assert isinstance(parameter_type, dict)
        assert len(parameter_type) == 1

        self.transformation_name = parameter_type.keys()[0]
        assert isinstance(self.transformation_name, str)

        super(AffineTransformation, self).__init__(parameter_type)

        self.parameter_space = CubicParameterSpace(parameter_type, min, max, ranges)

    def diffusion(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        det = np.linalg.det(mu)
        assert not det == 0

        A_inv = np.linalg.inv(mu)

        res = A_inv.dot(A_inv.T)
        # FIXME det is not handled on RHS
        #res *=np.abs(det)

        return res.reshape((1, 2, 2)).repeat(x.shape[0], axis=0)

    def transform(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        assert len(x.shape) == 2
        return np.einsum("ij,ej->ei", mu, x)

    def jacobian(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        res = mu.reshape((1, 2, 2)).repeat(x.shape[0], axis=0)
        return res

    def jacobian_inverse(self, x, mu=None):
        mu = mu[self.transformation_name] if isinstance(mu, Parameter) else mu
        return np.linalg.inv(self.jacobian(x, mu))

    def bounding_box(self, domain, mu):
        assert isinstance(mu, Parameter) or isinstance(mu, tuple) and all(isinstance(m, Parameter) for m in mu)
        assert isinstance(domain, np.ndarray)

        ll = domain[0,:]
        lr = np.array([domain[1,0], domain[0,1]])
        ul = np.array([domain[0,0], domain[1,1]])
        ur = domain[1,:]

        box = np.array([ll, lr, ul, ur])

        mu = (mu,) if isinstance(mu, Parameter) else mu

        box_transformed = np.array([self.transform(box, m) for m in mu])

        max_x = box_transformed[:,:,0].max(axis=(0,1))
        max_y = box_transformed[:,:,1].max(axis=(0,1))
        min_x = box_transformed[:,:,0].min(axis=(0,1))
        min_y = box_transformed[:,:,1].min(axis=(0,1))

        max = box_transformed.max(axis=(0,1))
        min = box_transformed.min(axis=(0,1))

        box_1 = np.array([[min_x, min_y],[max_x, max_y]])
        box_2 = np.array([min, max])

        return box_1

num_intervals = (100, 100)
domain = np.array([[0, 0], [1, 1]])


mu = {'transformation': np.array([[2, 0], [0, 2]])}
mu = Parameter(mu)

#functions
d_f_ = lambda X: -X[...,1]**2+X[...,1]
d_f_ = lambda X: X[...,1]
d_f = GenericFunction(d_f_, 2)

rhs = ConstantFunction(0, 2)


problem = EllipticProblem(rhs=rhs, dirichlet_data=d_f, name="elliptic")

transformation_type = {'transformation': (2,2)}
transformation = AffineTransformation(transformation_type, 0.5, 1.5)

trafo_problem = AffineTransformationProblem(problem, transformation)

elliptic_grid = TriaGrid(num_intervals)
trafo_grid = DomainTransformationTriaGrid(elliptic_grid, transformation)








# unstructured grid
vertices = trafo_grid.centers(2, mu)
faces = elliptic_grid.subentities(0, 2)

unstr_grid = UnstructuredTriangleGrid(vertices, faces)

bi_t = AllDirichletBoundaryInfo(trafo_grid)
bi = AllDirichletBoundaryInfo(unstr_grid)

discretization, _ = discretize_elliptic_cg(trafo_problem, grid=trafo_grid, boundary_info=bi_t)
discretization_u, _ = discretize_elliptic_cg(problem, grid=unstr_grid, boundary_info=bi)

discretization.disable_caching()
discretization_u.disable_caching()

U = discretization.solve(mu)
U_u = discretization_u.solve()
ERR = U - U_u

ERR_abs = ERR.copy()
ERR_abs._array = np.abs(ERR_abs._array)

discretization.visualize((U, U_u, ERR, ERR_abs),
                         mu=(mu,mu,mu,mu),
                         legend=("FEM", "Unstructured", "ERR", "abs(ERR)"),
                         separate_colorbars=True)

plt.triplot(vertices[..., 0], vertices[..., 1], faces)
plt.show()
z=0