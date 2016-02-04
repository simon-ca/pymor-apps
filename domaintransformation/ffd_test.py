import numpy as np
import matplotlib.pyplot as plt

from pymor.grids.tria import TriaGrid
from pymor.grids.unstructured import UnstructuredTriangleGrid

from pymor.domaindescriptions.basic import RectDomain

from pymor.analyticalproblems.elliptic import EllipticProblem

from pymor.discretizers.elliptic import discretize_elliptic_cg

from pymor.grids.boundaryinfos import AllDirichletBoundaryInfo
from pymor.parameters.base import Parameter

from pymor.parameters.spaces import CubicParameterSpace

#from domaintransformation.functions.basic import DomainTransformationFunction
#from domaintransformation.analyticalproblems.elliptic_transformation import TransformationProblem
#from domaintransformation.grids.domaintransformation import DomainTransformationTriaGrid
#from domaintransformation.discretizers.elliptic import discretize_elliptic_cg

from domaintransformation.functions.ffd import FreeFormDeformation

from domaintransformation.discretizers.transformation import discretize_elliptic_cg_ei

def parameter_mask(num_control_points, active=[]):
    assert isinstance(num_control_points, tuple)
    assert len(num_control_points) == 2

    K, L = num_control_points

    assert isinstance(active, list)
    assert len(active) <= K*L*2

    assert all(isinstance(a, tuple) for a in active)
    assert all(len(a) == 3 for a in active)
    assert all(isinstance(a, int) and isinstance(b, int) and isinstance(c, int) for a, b, c in active)
    assert all(x >= 0 and x < K and y >= 0 and y < L and z in [0, 1] for x, y, z in active)

    mask = np.zeros((K, L, 2), dtype=bool)
    for x,y,z in active:
        mask[x,y,z] = True

    return mask


domain = np.array([[0, 0], [1, 1]])
shift_min = -0.5
shift_max = 0.5
PROBLEM_NUMBER = 1
num_control_points = [(2, 2), (5,3), (5,3)][PROBLEM_NUMBER]
active = [[(0, 1, 0), (1, 1, 0)],
          [(4, 1, 0), (0, 2, 0)],
          [(0, 2, 1), (1, 2, 1), (2, 2, 1), (3, 2, 1), (4, 0, 0), (4, 1, 0), (4, 2, 0), (4, 2, 1)]
         ][PROBLEM_NUMBER]
mask = parameter_mask(num_control_points, active)

param = [np.array([1.0, 1.0]),
         np.array([0.15, 0.15]),
         np.array([0.15, -0.15, 0.15, -0.15, 0.15, -0.25, 0.15, 0.15])]
mu = {'ffd': param[PROBLEM_NUMBER]}
mu = Parameter(mu)

problem = EllipticProblem(name="elliptic")

transformation_type = {'transformation': (2,2)}
transformation = FreeFormDeformation(RectDomain(domain), mask, "ffd", shift_min, shift_max)
#rhs_transformation = FFDRHSTransformation(RectDomain(domain), mask, "ffd", shift_min, shift_max)
#trafo_problem = TransformationProblem(problem, transformation, rhs_transformation)


def run(N, plot=True, triplot=False):
    print(N)
    num_intervals = (N, N)
    elliptic_grid = TriaGrid(num_intervals)
    bi = AllDirichletBoundaryInfo(elliptic_grid)

    discretizations = discretize_elliptic_cg_ei(problem, transformation, grid=elliptic_grid, boundary_info=bi,
                                                options=[None, "mceim"])

    discretization, data = discretizations[0]
    discretization_eim , data_eim = discretizations[1]

    grid = data['grid']

    d_diff = discretization.operator.diffusion_function(grid.centers(0), mu=mu)
    d_diff_ei = discretization_eim.operator.diffusion_function(grid.centers(0), mu=mu)

    d_diff_ = d_diff - d_diff_ei
    y = np.allclose(d_diff, d_diff_ei)
    z = 0


    #U = discretization.solve(mu=mu)
    #discretization.visualize(U, mu=mu)

    trafo_grid = data['transformation_grid']

    # unstructured grid
    #vertices = elliptic_grid.centers(2)
    vertices_t = trafo_grid.centers(2, mu)
    faces = elliptic_grid.subentities(0, 2)

    unstr_grid = UnstructuredTriangleGrid(vertices_t, faces)

    #bi_t = AllDirichletBoundaryInfo(trafo_grid)
    bi_u = AllDirichletBoundaryInfo(unstr_grid)

    #discretization, _ = discretize_elliptic_cg(trafo_problem, grid=trafo_grid, boundary_info=bi_t)
    discretization_u, _ = discretize_elliptic_cg(problem, grid=unstr_grid, boundary_info=bi_u)

    discretization.disable_caching()
    discretization_u.disable_caching()
    discretization_eim.disable_caching()

    U = discretization.solve(mu)
    U_u = discretization_u.solve()
    U_eim = discretization_eim.solve(mu)

    ERR = U - U_u
    ERR_abs = ERR.copy()
    ERR_abs._array = np.abs(ERR_abs._array)

    ERR_eim = U_eim - U_u
    ERR_eim_abs = ERR_eim.copy()
    ERR_eim_abs._array = np.abs(ERR_eim_abs._array)

    ERR_u_eim = U - U_eim
    ERR_u_eim_abs = ERR_u_eim.copy()
    ERR_u_eim_abs._array = np.abs(ERR_u_eim_abs._array)

    if plot:
        discretization.visualize((U, U_u, ERR, ERR_abs),
                                 mu=(mu,mu,mu, mu),
                                 legend=("Trafo", "Unstructured", "ERR", "abs(ERR)"),
                                 separate_colorbars=True,
                                 title="Grid {}x{}".format(*num_intervals))
        discretization.visualize((U_eim, U_u, ERR_eim, ERR_eim_abs),
                                 mu=(mu,mu,mu, mu),
                                 legend=("EI", "Unstructured", "ERR", "abs(ERR)"),
                                 separate_colorbars=True,
                                 title="Grid {}x{}".format(*num_intervals))
        discretization.visualize((U, U_eim, ERR_u_eim, ERR_u_eim_abs),
                                 mu=(mu,mu,mu, mu),
                                 legend=("Trafo", "EI", "ERR", "abs(ERR)"),
                                 separate_colorbars=True,
                                 title="Grid {}x{}".format(*num_intervals))
    if triplot:
        pass
        #plt.triplot(vertices_t[..., 0], vertices_t[..., 1], faces)
        #plt.show()

    err_l2 = discretization.l2_norm(ERR_eim)
    err_h1_semi = discretization.h1_semi_norm(ERR_eim)
    err_h1 = discretization.h1_norm(ERR_eim)
    return {"ERR_L2": err_l2, "ERR_H1_SEMI": err_h1_semi, "ERR_H1": err_h1}

MIN = 100
MAX = 100
ERRS_L2 = []
ERRS_H1_SEMI = []
ERRS_H1 = []
NS = []
for N in range(MIN, MAX+1, 20):
    pass
    NS.append(N)
    r = run(N)
    ERRS_L2.append(r["ERR_L2"])
    ERRS_H1_SEMI.append(r["ERR_H1_SEMI"])
    ERRS_H1.append(r["ERR_H1"])



print(NS)
print(ERRS_L2)

N_ = np.linspace(MIN,MAX,100)
E_1 = 1.0/N_
E_2 = 1.0/(N_**2)




plt.plot(NS, ERRS_L2, label="ERR")
plt.plot(N_, E_1, label="10E-1")
plt.plot(N_, E_2, label="10E-2")
plt.xlabel("Number of Intervals")
plt.ylabel("L2 Error")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

plt.plot(NS, ERRS_H1,label="ERR")
plt.plot(N_, E_1, label="10E-1")
plt.plot(N_, E_2, label="10E-2")

plt.xlabel("Number of Intervals")
plt.ylabel("H1 Error")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()


#run(100, True, True)

