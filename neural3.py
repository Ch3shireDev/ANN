import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob

def nonlin(x,deriv=False):
	return x*(1-x) if deriv else 1/(1+np.exp(-x))
    
def open_im(letter, num):
    image_path = glob.glob("./%s/hsf_0/*.png" % letter)[num]
    image = imageio.imread(image_path)
    image = 1-np.sum(image, axis=2)/3/255
    image = image.reshape(-1)
    return image


x = []
y = []

for i in range(10):
    x += [open_im('A', i)]
    y += [[1,0]]

for i in range(10):
    x += [open_im('B', i)]
    y += [[0,1]]

x = np.array(x)
y = np.array(y)

l0 = x

num_inputs = 128**2
num_outputs = 2
from random import randint
HL = [num_inputs, 10, num_outputs]
bias = [True, True]

syn = [2*np.random.random((HL[i]+(1 if bias[i] else 0), HL[i+1])) - 1 for i in range(len(HL)-1)]
error = 0.5


zero = np.array([np.ones_like(x.T[0])]).T

while error > 0.01:

    L = [x]
    for i in range(len(HL)-1):
        if bias[i]:
            L[-1] = np.concatenate([L[-1], zero], axis=1)
        L += [nonlin(np.dot(L[i], syn[i]))]
    Y = L[-1]

    d_syn = [None for i in range(len(HL)-1)]

    error = y - L[-1]

    delta = error*nonlin(L[-1], deriv=True)
    d_syn[-1] = L[-2].T.dot(delta)
    error = delta.dot(syn[-1].T)

    for i in range(len(HL)-2):
        k = len(HL) - 2 - i
        r = k - 1
        t = -1 if bias[k] else 0
        delta = error*nonlin(L[k], deriv=True)
        if bias[k]:
            d_syn[r] = L[r].T.dot(delta.T[:-1].T)
            error = delta.T[:-1].T.dot(syn[r].T)
        else:
            d_syn[r] = L[r].T.dot(delta)
            error = delta.dot(syn[r].T)

    for i in range(len(HL)-1):
        syn[i] += d_syn[i]

    error = np.mean(np.abs(y - L[-1]))
    print(error)


x = np.array([open_im('B',1000)])
zero = np.array([np.ones_like(x.T[0])]).T
L = [x]
for i in range(len(HL)-1):
    if bias[i]:
        L[-1] = np.concatenate([L[-1], zero], axis=1)
    L += [nonlin(np.dot(L[i], syn[i]))]
Y = L[-1]
print('B:', Y)


x = np.array([open_im('A',1000)])
zero = np.array([np.ones_like(x.T[0])]).T
L = [x]
for i in range(len(HL)-1):
    if bias[i]:
        L[-1] = np.concatenate([L[-1], zero], axis=1)
    L += [nonlin(np.dot(L[i], syn[i]))]
Y = L[-1]
print('A:', Y)
