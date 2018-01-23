import numpy as np
import pickle as p
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn
from neural_network import Network

network = Network([1,10,10,10,1], './data/x')
network.load()

x = [[x] for x in np.linspace(0,1,10)]
y = [[np.random.random()] for _ in range(10)]

X = [xx[0] for xx in x]
Y = [yy[0] for yy in y]

plt.plot(X,Y, 'ro')

for ii in range(100):
    c = network.retarded_training(x,y)
    print(ii, c)

print(network.W)

network.save()

N = 128

X = np.linspace(0,1,N)
Y = []

for x in X:
    Y += [network.forward(x)[0]]

plt.plot(X,np.array(Y))

plt.show()