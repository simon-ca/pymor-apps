from pymor.functions.basic import FunctionBase
from pymor.parameters.base import Parameter

import numpy as np
import matplotlib.pyplot as plt

from domaintransformation.algorithms.ei import interpolate_function_analytically


class Example(FunctionBase):

    shape_range = tuple()
    dim_domain = 1

    def __init__(self):
        self.build_parameter_type({'exponent': 1}, local_global=True)

    def evaluate(self, x, mu=None):
        #mu = self.parse_parameter(mu)
        mu = mu['exponent']
        #return x**2*mu
        c = np.cos(3*np.pi*mu*(x+1))
        e = np.exp(-(1+x)*mu)
        return (1-x)*c*e


def demo(N, N_ei, mus, plot=True):

    X = np.linspace(-1,1,N)
    mu = 1

    f = Example()

    f_ei = interpolate_function_analytically(f, mus, X, 1.0E-10, 25)

    Y = f.evaluate(X, {'exponent': 1.25})
    Y_ei = f_ei.evaluate(X, {'exponent': 1.25})

    print(Y)
    print(Y_ei)

    plt.plot(X,Y,label="f")

    plt.legend()
    plt.show()


mus = np.linspace(1, 2, 6)
mus = [Parameter({'exponent': mu}) for mu in mus]
demo(100, 100, mus)