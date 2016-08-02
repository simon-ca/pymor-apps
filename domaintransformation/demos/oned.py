from pymor.functions.basic import FunctionBase
from pymor.parameters.base import Parameter

import numpy as np
import matplotlib.pyplot as plt

from domaintransformation.algorithms.ei import interpolate_function


class Example(FunctionBase):

    shape_range = tuple()
    dim_domain = 1

    def __init__(self):
        self.build_parameter_type({'exponent': 1}, local_global=True)

    def evaluate(self, x, mu=None):
        mu = mu['exponent']
        #return x**2*mu
        c = np.cos(3*np.pi*mu*(x+1))
        e = np.exp(-(1+x)*mu)
        #c = np.sin(3*np.pi*mu*(x+1))
        #e = np.exp(-(1-x)*mu)
        return (1-x)*c*e



def plot_solutions(f, x, mus):
    plt.figure("Solutions")
    for mu in mus:
        y = f(x,mu)
        m = mu['exponent']
        label = "$\mu={}$".format(m)
        plt.plot(x, y, label=label)

    plt.legend()
    plt.xlabel("$x$", fontsize=16)
    plt.ylabel("$f(x;\mu)$", fontsize=16)

    z = 0

def evaluate_eim(f, x, mu_train, mu_test, sizes):

    y = [f(x, mu) for mu in mu_test]
    y = np.array(y)

    mu_test = [mu['exponent'] for mu in mu_test]

    f_eis = []
    for size in sizes:
        f_ei, f_ei_data = interpolate_function(f, mu_train, x, None, size)
        f_eis.append((f_ei, f_ei_data))

    mus_taken = [mu_test[index] for index in f_ei_data['max_err_indices']]
    mus_taken = np.array(mus_taken, dtype=np.float)

    errs = []
    for i in range(len(f_eis)):
        f_ei, f_ei_data = f_eis[i]

        y_eim = [f_ei.evaluate(x, float(mu)) for mu in mu_test]
        y_eim = np.array(y_eim)

        d = y-y_eim

        e = np.max(np.abs(d), axis=1)
        errs.append(e)

    return np.array(errs), y, mus_taken

def plot_errors(sizes, errors, ys, relative=False):
    assert len(sizes) == len(errors)

    mi = errors.min(axis=1)
    me = errors.mean(axis=1)
    ma = errors.max(axis=1)

    error_type = "Relative" if relative else "Absolute"
    n_params = ys.shape[0]

    title = "{} errors for {} random parameters".format(error_type, n_params)

    if relative:
        y_max = ys.max()

        mi /= y_max
        me /= y_max
        ma /= y_max

    plt.figure(title)

    plt.semilogy(sizes, mi, label="min")
    plt.semilogy(sizes, me, label="mean")
    plt.semilogy(sizes, ma, label="max")

    plt.legend()

    plt.xlabel("$M$", fontsize=16)

    if relative:
        y_label = "$e_{rel}$"
    else:
        y_label = "$|e_{abs}|$"

    plt.ylabel(y_label, fontsize=16)


    z = 0


def plot_parameter_distribution(mus):
    plt.figure("Parameter distribution")
    plt.scatter(mus, np.zeros_like(mus))

    z = 0


def plot_basis_functions(f_eim, x):
    funcs = f_eim.operators_linear

    plt.figure("basis functions")
    for i, func in enumerate(funcs):
        label = "$q_{}$".format(i+1)
        plt.plot(x, func(x), label=label)

    plt.legend()
    plt.xlabel("$x$", fontsize=16)
    plt.ylabel("$q(x)$", fontsize=16)

    z = 0



def demo(f, x, mus, plot=True):

    mu = 1

    f_ei, _ = interpolate_function(f, mus, x, 1.0E-10, 30)

    Y = f.evaluate(x, {'exponent': 1.25})
    Y_ei = f_ei.evaluate(x, {'exponent': 1.25})

    print(Y)
    print(Y_ei)

    print(Y.shape)
    print(Y_ei.shape)

    plt.plot(x, Y, label="f")
    plt.plot(x, Y_ei, label="f_ei")

    plt.legend()
    plt.show()

f = Example()
x = np.linspace(-1, 1, 250+1)

mus = np.linspace(1, 3, 100)
mus = [Parameter({'exponent': mu}) for mu in mus]

#demo(f=f, x=x, mus=mus)

plotmus = [1.0, 1.5, 2.0]
plotmus = [Parameter({'exponent': mu}) for mu in plotmus]
#plot_solutions(f, x, plotmus)

sizes = range(1, 31)

import random
testmus = [random.uniform(1, 3) for i in range(250)]
testmus = [Parameter({'exponent': mu}) for mu in testmus]

#errs, ys, mus_taken = evaluate_eim(f, x, mu_train=mus, mu_test=testmus, sizes=sizes)

#plot_errors(sizes, errs, ys, relative=False)
#plot_errors(sizes, errs, ys, relative=True)

#plot_parameter_distribution(mus_taken)

N = 3
f_ei, _ = interpolate_function(f, mus, x, None, N)
plot_basis_functions(f_ei, x)

z = 0