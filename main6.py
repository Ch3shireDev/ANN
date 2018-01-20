import numpy as np
from numpy.random import randn, randint, uniform
import pickle as p
import sys
from plot import plot
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def S(y1,y2):
    return sum((y1-y2)**2)/2

def cost(x, y, w):
    yhat = forward(x,w)
    yy = sum((yhat-y).T**2/2)
    return yy

def forward(x,w):
    return sigmoid(w[0]+w[1]*x)

n = 1024

x = np.linspace(0,1,n)
w = [0,0]
y = np.array([0 if i<n/2 else 1 for i in range(n)])


def opt(x,y,w,I):
    w0, w1 = -10, 100
    n = 8
    dw = (w1-w0)/n
    cmin = 0

    for i in range(20):
        wmin = np.linspace(w0, w1, n+1)

        c = []
        for ww in wmin:
            if I==0:  
                c += [cost(x, y, [ww,w[1]])]
            elif I==1:
                c += [cost(x, y, [w[0],ww])]

        c = np.array(c)

        index = np.argmin(c)
        wmin = wmin[index]
        cmin = c[index]
        dw = (w1-w0)/n
        w0, w1 = wmin-dw, wmin+dw


        if dw<0.0001:
            break
    w[I] = wmin
    return cmin


for i in range(1000):
    c = opt(x,y,w,0)
    c = opt(x,y,w,1)
    print(c, w)

plt.plot(x,y)
plt.plot(x,forward(x,w))
plt.show()