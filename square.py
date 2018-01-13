import numpy as np
from numpy.random import randn
import pickle as p
from plot import plot

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidPrime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

def error(x, y):
    return np.sum((x-y)**2)/2

def costFunction(X, Y):
    yHat = forward(X)
    J = sum((Y-yHat)**2)/2
    return J[0]

def forward(X):
    global W1, W2, W3
    x1 = np.dot(X, W1)
    x1 = sigmoid(x1)
    x2 = np.dot(x1, W2)
    x2 = sigmoid(x2)
    x3 = np.dot(x2, W3)
    x3 = sigmoid(x3)
    return x3

def costFunctionPrime(X, y):
    yHat = forward(X)
    
    z2 = np.dot(X, W1)
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W2)
    a3 = sigmoid(z3)

    z4 = np.dot(a3, W3)

    delta3 = np.multiply(-(y-yHat), sigmoidPrime(z4))
    dJdW3 = np.dot(a3.T, delta3)

    delta2 = np.dot(delta3, W3.T)*sigmoidPrime(z3)
    dJdW2 = np.dot(X.T, delta2)  
    
    delta1 = np.dot(delta2, W2.T)*sigmoidPrime(z2)
    dJdW1 = np.dot(X.T, delta1)  
    

    return dJdW1, dJdW2, dJdW3

fname = "square.dat"

Nin = 1
Nh = 3
N2 = 3
Nout = 1

N = [1,3,3,1]


fname = "square"+('-'.join([str(s) for s in N]))+".dat"

W1, W2, W3 = None, None, None
W = None

try:
    f = open(fname, 'rb')
    W1, W2, W3 = p.load(f)
    f.close()
    if W1.shape != (Nin,Nh) or W2.shape != (Nh, N2) or W3.shape != (N2, Nout):
        raise Exception
except:
    print("Error:", fname, "corrupted or missing, creating new synapse data...")
    W1 = randn(Nin, Nh)
    W2 = randn(Nh, N2)
    W3 = randn(N2, Nout)


X = np.linspace(0,1,50).reshape(50,1)
Y = X**2

err = error(Y, forward(X))
err = "%0.03f" % (err,)
print(err)


scalar = 0.1

for i in range(10000):
    dJdW1, dJdW2, dJdW3 = costFunctionPrime(X, Y)
    W1 -= scalar*dJdW1
    W2 -= scalar*dJdW2
    W3 -= scalar*dJdW3
    cost3 = costFunction(X, Y)
    if i%100==0:
        print(i, cost3)


f = open(fname, 'wb')
p.dump((W1, W2, W3), f)
f.close()

Yhat = forward(X)

import matplotlib.pyplot as plt

plt.plot(X, Yhat)
plt.plot(X, Y)
plt.show()