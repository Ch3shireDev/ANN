import numpy as np
import matplotlib.pyplot as plt

def nonlin(x,deriv=False):
	if deriv:
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([np.linspace(0, 1, 128)]).T   
y = np.sin(X*np.pi*4)/2+0.5

l0 = X
l0 = np.array([np.array(X.T[0]), np.ones(X.shape[0])]).T

num_inputs = 1
num_outputs = 1
HL = [num_inputs+1, 32, 32, num_outputs]

def new_syn():
    syn = []
    for i in range(len(HL)-1):
        syn += [2*np.random.random((HL[i],HL[i+1])) - 1]
    return syn

def get_lin(syn):
    L = [l0]

    for i in range(len(HL) - 1):
        L += [nonlin(np.dot(L[i],syn[i]))]

    return L

def get_back_prop(y, L):

    d_syn = []

    error = y - L[-1]
    
    
    for i in range(len(HL) - 1):
        delta = error*nonlin(L[len(HL) - 1 - i], deriv=True)
        d_syn = [L[len(HL) - 2 - i].T.dot(delta)] + d_syn
        error = delta.dot(syn[len(HL) - 2 - i].T)

    return d_syn


syn = new_syn()

error = 0.5

flag = True

while flag:
    flag = False        
    for i in range(1000):
        L = get_lin(syn)
        Y = L[-1]

        d_syn = get_back_prop(y, L)

        for i in range(len(HL) - 1):
            syn[i] += d_syn[i]

    error = np.mean(np.abs(y - L[-1]))

    print("Error:", str(error))

    if error>0.45:
        syn = new_syn()
        flag = True
    if error>0.01:
        flag = True
    
plt.plot(X, y)
plt.plot(X, Y)
plt.show()