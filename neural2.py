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
HL = [num_inputs, 8, 8, num_outputs]


syn = []
syn += [2*np.random.random((HL[0] + 1, HL[1])) - 1]
syn += [2*np.random.random((HL[1] + 1, HL[2])) - 1]
syn += [2*np.random.random((HL[2], HL[3])) - 1]
error = 0.5


while True:
    L = [add_one(l0)]

    L += [nonlin(np.dot(L[0], syn[0]))]

    L[-1] = np.concatenate([L[-1].T, np.array([np.ones(128)])]).T

    L += [nonlin(np.dot(L[1], syn[1]))]
    L += [nonlin(np.dot(L[2], syn[2]))]

    Y = L[-1]


    d_syn = []

    error = y - L[-1]
    
    delta = error*nonlin(L[3], deriv=True)
    d_syn = [L[2].T.dot(delta)] + d_syn
    error = delta.dot(syn[2].T)

    delta = error*nonlin(L[2], deriv=True)
    d_syn = [L[1].T.dot(delta)] + d_syn
    error = delta.dot(syn[1].T).T[:-1].T

    delta = error*nonlin(L[1].T[:-1].T, deriv=True)
    d_syn = [L[0].T.dot(delta)] + d_syn
    # error = delta.T[:-1].T.dot(syn[0].T)
    error = delta.dot(syn[0].T)


    syn[0] += d_syn[0]
    # syn[0] += d_syn[0].T[:-1].T
    syn[1] += d_syn[1]
    syn[2] += d_syn[2]

    error = np.mean(np.abs(y - L[-1]))
    
    print(error)
    if error<0.01:
        break


print(error)
plt.plot(y)
plt.plot(Y)
plt.show()