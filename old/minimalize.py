import matplotlib.pyplot as plt
import numpy as np


n = 16

i = 0
def f(x):
    global i
    i += 1
    return -np.exp(-(x-15.342873)**2/1000)+np.cos(x/10)/10*np.exp(-x**2/10000)

x0, x1 = -100, 100
# n = 8
dx = (x1-x0)/n

for _ in range(20):
    x = np.linspace(x0, x1, n+1)
    y = f(x)
    index = np.argmin(y)
    xmin = x[index]
    dx = (x1-x0)/n
    x0, x1 = xmin-dx, xmin+dx
    print(i,xmin,dx)

    if dx<0.000001:
        break

print(i)

x = np.linspace(-100,100,2**15)
y = f(x)

plt.plot(x,y)
plt.plot(xmin,f(xmin),'ro')
plt.show()