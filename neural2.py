import numpy as np
import matplotlib.pyplot as plt

def nonlin(x,deriv=False):
	return x*(1-x) if deriv else 1/(1+np.exp(-x))
    
def add_one(X):
    return np.array([np.array(X.T[0]), np.ones(X.shape[0])]).T

X = np.array([np.linspace(0, 1, 128)]).T   
y = np.sin(X*np.pi*4)/2+0.5

l0 = X

num_inputs = 1
num_outputs = 1
HL = [num_inputs, 8, num_outputs]


syn = []
syn += [2*np.random.random((HL[0] + 1, HL[1])) - 1]
syn += [2*np.random.random((HL[1] + 1, HL[2])) - 1]
error = 0.5


for i in range(1000):
    L = [add_one(l0)]
    a = syn[0]
    a = nonlin(np.dot(L[0], a))

    b = np.array([np.ones(X.shape[0])]).T
    a = np.concatenate((a,b),axis=1)

    L += [a]
    L += [nonlin(np.dot(L[1], syn[1]))]
    Y = L[-1]


    d_syn = []

    error = y - L[-1]
    
    delta = error*nonlin(L[2], deriv=True)
    d_syn = [L[1].T.dot(delta)] + d_syn
    error = delta.dot(syn[1].T)

    delta = error*nonlin(L[1], deriv=True)
    d_syn = [L[0].T.dot(delta)] + d_syn
    error = delta.dot(syn[0].T)

    syn[0] += d_syn[0]
    syn[1] += d_syn[1]

error = np.mean(np.abs(y - L[-1]))

print(error)
plt.plot(X, y)
plt.plot(X, Y)
plt.show()