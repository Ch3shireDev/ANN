import numpy as np
import pickle as p
from plot import plot

def sigmoid(x):
    return 1/(1+np.exp(-x))

def error(x, y):
    return np.sum((x-y)**2)/2

fname = "brain.dat"

Nin = 2
Nh = 3
Nout = 1


fname = "brain"+('-'.join([str(s) for s in [Nin,Nh,Nout]]))+".dat"

W1, W2 = None, None


try:
    f = open(fname, 'rb')
    W1, W2 = p.load(f)
    f.close()
    if W1.shape != (Nin,Nh) or W2.shape != (Nh,Nout):
        raise Exception
except:
    print("Error:", fname, "corrupted or missing, creating new synapsis data...")
    W1 = np.random.randn(Nin, Nh)
    W2 = np.random.randn(Nh, Nout)


X = np.array(([3,5], [5,1], [10,2]), dtype=float)
Y = np.array(([75], [82], [93]), dtype=float)

X = X/np.amax(X, axis=0)
Y = Y/100 #Max test score is 100

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

plot(forward, X, Y, err)