import numpy as np
from numpy.random import randn, randint, uniform
import pickle as p
import sys
from plot import plot

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

def error(x, y):
    return np.sum((x-y)**2)/2

def costFunction(X, Y, W):
    yHat = forward(X, W)
    J = sum((Y-yHat)**2)/2
    return J[0]

def forward(X, W):
    x1 = np.dot(X, W[0])
    x1 = sigmoid(x1)
    x2 = np.dot(x1, W[1])
    x2 = sigmoid(x2)
    x3 = np.dot(x2, W[2])
    x3 = sigmoid(x3)
    return x3

def costFunctionPrime(X, Y, W):
    yHat = forward(X, W)
    
    z2 = np.dot(X, W[0])
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W[1])
    a3 = sigmoid(z3)

    z4 = np.dot(a3, W[2])

    delta3 = np.multiply(-(Y-yHat), sigmoidPrime(z4))
    dJdW3 = np.dot(a3.T, delta3)

    delta2 = np.dot(delta3, W[2].T)*sigmoidPrime(z3)
    dJdW2 = np.dot(X.T, delta2)  
    
    delta1 = np.dot(delta2, W[1].T)*sigmoidPrime(z2)
    dJdW1 = np.dot(X.T, delta1)  
    

    return np.array([dJdW1, dJdW2, dJdW3])

N = [1,20,10,1]

fname = "half"+('-'.join([str(s) for s in N]))+".dat"

W = None

try:
    f = open(fname, 'rb')
    W = p.load(f)
    f.close()

    print(W.shape)

    if W[0].shape != (N[0],N[1]) or W[1].shape != (N[1], N[2]) or W[2].shape != (N[2], N[3]):
        raise Exception
except:
    print("Error:", fname, "corrupted or missing, creating new synapse data...")
    W1 = randn(N[0], N[1])
    W2 = randn(N[1], N[2])
    W3 = randn(N[2], N[3])
    W = np.array([W1,W2,W3])

    f = open(fname, 'wb')
    p.dump(W, f)
    f.close()


X = np.linspace(0,1,50).reshape(50,1)
Y = X/2

err = error(Y, forward(X, W))
err = "%0.03f" % (err,)
print(err)


scalar = -1
if len(sys.argv) > 1:
    scalar = float(sys.argv[1])
    print(scalar)

c0, c1 = -1,-1

for i in range(100000):
    dJdW = costFunctionPrime(X, Y, W)
    W += scalar*dJdW
    cost3 = costFunction(X, Y, W)

    if i%100==0:
        print(i, cost3)
        f = open(fname, 'wb')
        p.dump(W, f)
        f.close()



Yhat = forward(X, W)

import matplotlib.pyplot as plt

plt.plot(X, Yhat)
plt.plot(X, Y)
plt.show()