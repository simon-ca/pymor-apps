import numpy as np
from scipy.misc import comb
from matplotlib import pyplot as plt

x = np.linspace(0, 1, 100)

K = 4

ys = []

for k in range(K+1):
    y = comb(K, k, True)*(1-x)**(K-k)*x**k
    ys.append(y)

for k, y in enumerate(ys):
    label = "$b^{}_{}$".format(K, k)
    plt.plot(x, y, label=label)

plt.legend()

z = 0

