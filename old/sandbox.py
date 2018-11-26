import numpy as np
from random import randint

N = 2


x = [randint(0,300) for i in range(N)]

def f(xx):
    global x0
    s = 0

    x = []

    for i in range(len(xx)-1):
        x += [xx[i]-xx[i+1]]

    x+= [xx[-1]+xx[0]]

    for i in range(len(x)):
        s += (x[i]-2*i+1)*(x[i]-3*i+1+100)*(x[i]+4*i+1)*(x[i]-5*i+1)
    return s**2

n = 16

print(f(x))

for j in range(1):
    for i in range(N):
        xmin, xmax = -10000, 10000
        for k in range(10):
            x[i] = np.linspace(xmin, xmax, n+1)
            y = f(x)
            index = np.argmin(y)
            dx = (xmax-xmin)/n
            x[i] = x[i][index]
            xmin, xmax = x[i]-dx, x[i]+dx
    print(j, f(x))


