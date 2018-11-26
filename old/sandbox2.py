import numpy as np
from random import randint

N = 500
n = 8

x = np.array([0 for i in range(N)])
ex = []

for i in range(N):
    e = [0 for _ in range(N)]
    e[i] = 1
    ex += [e]

def f(x):
    X = None
    if len(x.shape)>1:
        X = np.outer(np.arange(1,N+1), np.ones(n+1))
    else: 
        X = np.arange(1,N+1)
    y = x - X + np.roll(x,1,axis=0)
    return sum(y**2)

print(f(x))

for i in range(100):
    for k in range(len(ex)):
        e = ex[k]
        e = np.array(e).astype(np.float)
        xmin, xmax = -100000, 100000
        x0 = x - e*np.dot(x, e)
        for j in range(15):
            lin = np.linspace(xmin,xmax,n+1)
            X = x0.T + np.outer(lin, e)
            y = f(X.T)
            index = np.argmin(y)
            x1 = lin[index]
            dx = (xmax-xmin)/n
            xmin, xmax = x1 - dx, x1 + dx   
        x = x0 + e*x1
    print(i, y[index])
    if y[index] < 5.1:
        break
