import numpy as np
from numpy.random import randn
import pickle as p
from plot import plot

def sigmoid(x):
    return 1/(1+np.exp(-x))

def error(x, y):
    return np.sum((x-y)**2)/2

fname = "square.dat"

Nin = 1
Nh = 3
Nout = 1


fname = "square"+('-'.join([str(s) for s in [Nin,Nh,Nout]]))+".dat"

W1, W2 = None, None


try:
    f = open(fname, 'rb')
    W1, W2 = p.load(f)
    f.close()
    if W1.shape != (Nin,Nh) or W2.shape != (Nh, Nout):
        raise Exception
except:
    print("Error:", fname, "corrupted or missing, creating new synapse data...")
    W1 = randn(Nin, Nh)
    W2 = randn(Nh, Nout)


X = np.linspace(0,1,50)
Y = X**2

def forward(X):
    x1 = np.dot(X, W1)
    x1 = sigmoid(x1)
    x2 = np.dot(x1, W2)
    x2 = sigmoid(x2)
    return x2

err = error(Y, forward(X))
err = "%0.03f" % (err,)
print(err)

f = open(fname, 'wb')
p.dump((W1, W2), f)
f.close()

Yhat = forward(X)

import matplotlib.pyplot as plt

plt.plot(X, Yhat)
plt.show()