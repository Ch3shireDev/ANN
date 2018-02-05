import numpy as np
import pickle as p
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn
from neural import network

net = network([1,3,1], filename='./data/aaa', bias=True)
net.load()

x = np.array([[x] for x in np.linspace(0,1,16)])
y = (np.sin(2*np.pi*x)+1)/2

# net.expand(1)

for i in range(1000):
    c = net.retarded_training(x,y)
    print(i,c,net.N)

net.save()

X = [[x] for x in np.linspace(0,1,1024)]
Y = net.forward(X)

plt.plot(X,Y)
plt.plot(x,y, 'ro')
plt.show()