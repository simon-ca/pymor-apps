from pymor.functions.interfaces import FunctionInterface

from domaintransformation.algorithms.ei import ei_greedy_function, interpolate_function_test, interpolate_function

import matplotlib.pyplot as plt

import numpy as np

from pymor.domaindescriptions.basic import LineDomain
from pymor.grids.oned import OnedGrid


class Example(FunctionInterface):

    shape_range = tuple()
    dim_domain = 1

    def __init__(self):
        self.build_parameter_type({'exponent': 1}, local_global=True)

    def evaluate(self, x, mu=None):
        mu = self.parse_parameter(mu)
        mu = mu['exponent']
        #return x**2*mu
        c = np.cos(3*np.pi*mu*(x+1))
        e = np.exp(-(1+x)*mu)
        return (1-x)*c*e


class Example2(FunctionInterface):

    shape_range = (2,2)
    dim_domain = 1

    def __init__(self):
        self.build_parameter_type({'exponent': 1}, local_global=True)

    def evaluate(self, x, mu=None):
        assert x.ndim == 1 or x.ndim == 0
        mu = self.parse_parameter(mu)
        mu = mu['exponent']

        a = np.array([[mu, -mu], [2, mu**2]]).astype(x.dtype)

        if x.ndim == 0:
            # one point
            x_ = np.array([[x, x], [1, x**2]])
        else:
            x_ = np.array([[x, x],[np.ones_like(x, x.dtype), x**2]])
            x_ = x_.swapaxes(0, 2).swapaxes(1, 2)
        arr = x_ * a.reshape((1,)+a.shape)
        arr = arr.astype(x.dtype)
        return arr
        #return (np.eye(2)*mu).reshape((1,2,2)).repeat(x.shape[0], axis=0)
        try:
            res = arr.reshape((1,)+arr.shape).repeat(x.shape[0], axis=0)
        except IndexError:
            res = arr.reshape((1,)+arr.shape)
        assert res.shape[-2:] == self.shape_range
        return res



def demo(N, N_ei, mus, plot=True):
    evaluations = []
    X = np.linspace(-1,1,N)
    X_ei = np.linspace(-1,1,N_ei)
    mu = 1

    f = Example()

    for mu in mus:
        Y = f.evaluate(X_ei, mu)
        #print(Y)

        if plot:
            plt.plot(X_ei,Y, label="mu = {}".format(mu))
        evaluations.append(Y)
    evaluations = np.array(evaluations)
    print(evaluations)

    f_ei = interpolate_function_test(f, mus, X_ei)

    Y = f.evaluate(X, 1.25)
    Y_ei = f_ei.evaluate(X, {'exponent': 1.25})

    print(Y)
    print(Y_ei)

    plt.plot(X,Y,label="f")
    plt.plot(X_ei,Y_ei, label="f_ei")
    plt.legend()
    plt.show()

def demo_2d(N, N_ei, mus, plot=True):
    evaluations = []
    X = np.linspace(-1,1,N)

    f = Example2()

    f_ei = interpolate_function_test(f, mus, X)

    mu = {'exponent': 2}

    test_mus = [{'exponent': mu} for mu in range(1, 20)]

    for test_mu in test_mus:
        print("Testing mu = {}".format(test_mu))
        z = f.evaluate(X, mu=test_mu)
        z_ei = f_ei.evaluate(X, mu=test_mu)

        assert z.shape == z_ei.shape

        print("z = {}".format(z))
        print("z_ei = {}".format(z_ei))
        print("z-z_ei = {}".format(z-z_ei))

        assert np.allclose(z, z_ei)

        print("allclose: {}".format(np.allclose(z, z_ei)))


    i = 0


def demo_pymor(N, N_mus, plot=True):
    domain = LineDomain()
    grid = OnedGrid(num_intervals=N)
    f = Example()
    type = "quadrature"
    X = grid.centers(0) if type == "center" else grid.quadrature_points(0, order=2)
    X = X.ravel()
    mu = {'exponent': 1.25}
    f_eis = []

    for N_mu in N_mus:
        mus = np.linspace(1, np.pi, N_mu)
        f_eis.append(interpolate_function(f, mus, grid, "quadrature", 1.0e-10, 25))



    Y = f.evaluate(X, mu)
    Y_eis = []
    for i in range(len(f_eis)):
        Y_eis.append(f_eis[i].evaluate(X, mu))
        plt.plot(X, Y_eis[-1], label="{}".format(N_mus[i]))

    plt.plot(X,Y, label="exact")
    plt.legend()
    plt.show()

    #print(f_ei)

def demo_2d_pymor(N, N_mus, plot=True):
    domain = LineDomain()
    grid = OnedGrid(num_intervals=N)
    f = Example2()
    type = "quadrature"
    X = grid.centers(0) if type == "center" else grid.quadrature_points(0, order=2)
    X = X.ravel()
    mu = {'exponent': 1.25}
    f_eis = []

    for N_mu in N_mus:
        mus = np.linspace(1, np.pi, N_mu)
        f_eis.append(interpolate_function(f, mus, grid, "quadrature", 1.0e-10, 25))



    Y = f.evaluate(X, mu)
    Y_eis = []
    for i in range(len(f_eis)):
        Y_eis.append(f_eis[i].evaluate(X, mu))

    assert all([np.allclose(Y, Y_ei) for Y_ei in Y_eis])

    print("ALLCLOSE: {}".format(all([np.allclose(Y, Y_ei) for Y_ei in Y_eis])))

    #print(f_ei)



demo(1001, 11, np.linspace(1, np.pi, 10), False)
demo_pymor(1001, [1,2,5,10,15])
demo_2d(1001, 11, np.arange(1, 10), False)
#demo_2d_pymor(1001, np.arange(1, 10), False)
