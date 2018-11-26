import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn
import scipy.special as sp


def f(x,y):
    s = 0
    for i in range(5):
        s+=(x-3.533)**2*np.cos(i*(y-14))**2+(y-22.441)**2
    return s

n = 8

x,y = 15,12

c1,c2 = 1,2

i = 0
# while abs(c2-c1)>1e-6:
for i in range(10):

    xmin, xmax = -100, 100
    ymin, ymax = -100, 100

    for _ in range(10):
        x = np.linspace(xmin,xmax,n+1)
        index = np.argmin(f(x,y))
        dx = (xmax-xmin)/n
        x = x[index]
        xmin, xmax = x - dx, x + dx

    for _ in range(10):
        y = np.linspace(ymin,ymax,n+1)
        index = np.argmin(f(x,y))
        dy = (ymax-ymin)/n
        y = y[index]
        ymin, ymax = y - dy, y + dy

    c1, c2 = c2, f(x,y)
    print(i, c2, x, y)


N = 128

X = np.linspace(-20,20,N)
Y = np.linspace(-20,20,N)
Z = []


for x in X:
    for y in Y:
        Z += [f(x,y)]


X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
Z = Z.reshape(N,N)

X, Y = np.meshgrid(X,Y)


fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(X,Y,Z.T)
plt.show()