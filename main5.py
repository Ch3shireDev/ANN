import numpy as np
from numpy.random import randn, randint, uniform
import pickle as p
import sys
from plot import plot
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

def error(x, y):
    return np.sum((x-y)**2)/2

def costFunction(X, Y, W):
    yHat = forward(X, W)
    J = sum((Y-yHat)**2)/2
    return J

def forward(X, W):
    x1 = np.outer(X,W)
    x1 = sigmoid(x1)
    return x1

def costFunctionPrime(X, Y, W):
    yHat = forward(X, W)




N = [1, 1]

W = randn(1,1)

x = np.linspace(0,1,1024)
y = np.concatenate([np.zeros(512), np.ones(512)])

N = [1,1]
W = None


fname = "./data/half"+('-'.join([str(s) for s in N]))+".dat"

try:
    f = open(fname, 'rb')
    W = p.load(f)
    f.close()

    print(W.shape)

    if W.shape != (N[0],N[1]):
        raise Exception
except:
    print("Error:", fname, "corrupted or missing, creating new synapse data...")
    W = randn(N[0], N[1])

    f = open(fname, 'wb')
    p.dump(W, f)
    f.close()


f = open(fname, 'wb')
p.dump(W, f)
f.close()


# for i in range(10000):
#     W[0][0] += W[0][0]/100
#     c1, c2 = c2, costFunction(x,y,W)
#     print(W[0][0],c1, c2)

# c = costFunction(x,y,W)

n= 16
x0, x1 = -100, 100
dx = (x1-x0)/n

# for _ in range(20):
w = np.linspace(x0, x1, n+1)

# y=np.dot(w,x.T)
print(y)

a = x
b = np.array([1,2])

c = np.outer(a,b) - np.outer(x,np.ones_like(b))

print(sum(c))


# forward(x, w)


# y = costFunction(x, y, w)
    # index = np.argmin(y)
    # xmin = x[index]
    # dx = (x1-x0)/n
    # x0, x1 = xmin-dx, xmin+dx
    # print(i,xmin,dx)

    # if dx<0.000001:
        # break


# f = open(fname, 'wb')
# p.dump(W, f)
# f.close()
