import numpy as np
import pickle as p
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn
from neural_network import network
from itertools import product

net = network([1,3,3,1], filename='./data/x')
net.load_random()

x = [[x] for x in np.linspace(0,1,4)]
y = [[np.random.random()] for _ in range(4)]

X = [xx[0] for xx in x]
Y = [yy[0] for yy in y]

dw = 0.001

# E = []
# for w in net.W:
#     E += [np.zeros_like(w)]
# E = np.array(E)

# indices = []
# for i in range(len(E)):
#     for j in range(len(E[i])):
#         for k in range(len(E[i][j])):
#             indices += [(i,j,k)]

# for ii in range(50):
#     print([np.linalg.norm(e) for e in E])
#     for kk in range(len(x)):
#         grad = E.copy()
#         x0, y0 = x[kk], y[kk]
#         y1 = net.cost(x0,y0,net.W)
#         for i,j,k in indices:
#             E[i][j][k] = 1
#             y2 = net.cost(x0,y0,net.W + E*dw)
#             grad[i][j][k] += (y2-y1)/dw
#             E[i][j][k] = 0

#         for i in range(len(grad)):
#             g = np.linalg.norm(grad[i])
#             if g>1e-6:
#                 grad[i] /= g
        
#         wmin,wmax = -100, 100
#         n = 8
#         for _ in range(10):
#             Ws = np.linspace(wmin,wmax,n+1)
#             c = []
#             for w in Ws:
#                 c += [net.cost(x0,y0,net.W+w*grad)]
#             index = np.argmin(c)
#             c = c[index]
#             w = Ws[index]
#             dw = (wmax-wmin)/n
#             wmin, wmax = w-dw, w+dw
#         net.W += w*grad
#     print(ii, c, [np.linalg.norm(w) for w in net.W], dw)

for i in range(50):
    net.retarded_training(x,y)

net.save()

N = 128

plt.plot(X,Y, 'ro')

X = np.linspace(0,1,N)
Y = []

for x in X:
    Y += [net.forward(x)[0]]

plt.plot(X,np.array(Y))
plt.show()

