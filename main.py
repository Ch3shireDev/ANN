import numpy as np
import pickle as p
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn
from neural_network import network
from itertools import product

net = network([1,10,10,1], filename='./data/x')
net.load_random()
# net.load()

x = [[x] for x in np.linspace(0,1,4)]
y = [[np.random.random()] for _ in range(4)]

X = [xx[0] for xx in x]
Y = [yy[0] for yy in y]


Wmin =-10
Wmax = 10
n=8

Wzero = net.W.copy()

for i in range(len(Wzero)):
    Wzero[i] = np.zeros_like(Wzero[i])

indices = []

for k in range(len(Wzero)):
    a, b = Wzero[k].shape
    for l, m in product(range(a), range(b)):
        indices += [(k,l,m)]

for ii in range(20):
    dW = Wzero.copy()
    Wout = Wzero.copy()
    for k, l, m in indices:
        W = net.W
        wmin = Wmin
        wmax = Wmax
        for i in range(10):
            Ws = np.linspace(wmin, wmax, n+1)
            C = []
            for w in Ws:
                W[k][l][m] = w
                costs = net.cost(x, y, W)

                # dW[k][l][m] = w
                # costs = net.cost(x,y,net.W+dW)
                # dW[k][l][m] = 0

                C += [sum(costs)]

            index = np.argmin(C)
            w = Ws[index]
            c = C[index]
            dw = (wmax - wmin)/n
            wmin, wmax = w - dw, w + dw

        # dW[k][l][m] = w
        # Wout[k][l][m] = w
        # Wout[k][l][m] += dW[k][l][m]

    net.W = np.array([w.copy() for w in W])

    print(ii,c,[np.linalg.norm(e) for e in net.W])

net.save()

N = 128

plt.plot(X,Y, 'ro')

X = np.linspace(0,1,N)
Y = []

for x in X:
    Y += [net.forward(x)[0]]

plt.plot(X,np.array(Y))
plt.show()


