import numpy as np
import pickle as p
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn
from neural_network import network

net = network([1,3,3,3,1], filename='./data/x')
# net.load_random()
net.load()

n = 32

x = np.array([[x] for x in np.linspace(0,1,n)])
y = (1+np.sin(10*x))/2

X = [xx[0] for xx in x]
Y = [yy[0] for yy in y]

plt.plot(x,y)

c = 1
ii = 0

for ii in range(1001):
    c = net.retarded_training(x,y)
    print(ii,c)
    if ii%100==0:
        net.shake(x,y)
        net.save()

N = 128

plt.plot(X,Y, 'ro')

X = np.linspace(0,1,N)
Y = []

for x in X:
    Y += [net.forward(x)[0]]

plt.plot(X,np.array(Y))
plt.show()


#  for i in range(len(self.z)):
#         if i==0:
#         yHat = self.forward(x)
#         delta = np.multiply(yHat - y, sigmoidPrime(self.z[-1]))
#         dJdW  = np.dot(self.a[-2].T, delta)
#     else:
#         delta = np.dot(delta, self.W[-i].T)*sigmoidPrime(self.z[-1-i])
#         dJdW = np.dot(self.a[-2-i].T, delta)

#     dJ += [dJdW]

# dJ = dJ[::-1]