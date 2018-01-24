from neural_network import network, sigmoid, sigmoidPrime
from copy import deepcopy
import numpy as np

x = np.array([[x] for x in np.linspace(0,1,20)])
y = np.sin((np.pi+1)*2*x)/2+1/2

net = network([1,3,3,3,1])
net.load_random()

net2 = deepcopy(net)

for ii in range(3):
    c = net.retarded_training(x,y)
    print(ii, c)

for ii in range(3):
    c = net2.retarded_training(x,y)
    print(ii, c)


def forward(self,X):
    z2 = np.dot(X,W1)
    a2 = sigmoid(z2)
    z3 = np.dot(a2,W2)
    yHat = sigmoid(z3)
    return yHat


def costFunction(self, X, y):
    #Compute cost for given X,y, use weights already stored in class.
    yHat = forward(X)
    J = 0.5*sum((y-yHat)**2)
    return J

def costFunctionPrime(x, y):
    yHat = forward(x)

    delta3 = np.multiply(-(y-yHat), sigmoidPrime(z3))
    dJdW2 = np.dot(a2.T, delta3)

    delta2 = np.dot(delta3, W2.T)*sigmoidPrime(z2)
    dJdW1 = np.dot(x.T, delta2)  
    
    return dJdW1, dJdW2


X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100

yHat = NN.forward(X)
scalar = 3

for i in range(10000):
    dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
    NN.W1 = NN.W1 - scalar*dJdW1
    NN.W2 = NN.W2 - scalar*dJdW2
    cost3 = NN.costFunction(X, y)
