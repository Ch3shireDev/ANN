import numpy as np
import pickle as p
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn

class Network:
    def __init__(self, N, fname):
        if(len(N)<2):
            print("Wrong size of N. len(N) should be greater than 1.")
            exit()
        self.N = N
        self.fname = None
        if fname is not None:
            self.fname = fname + ('-'.join([str(s) for s in N]))+".dat"

    def load_random(self):
        N = self.N
        self.W = [randn(N[i-1], N[i]) for i in range(1,len(N))]

    def save(self):
        f = open(self.fname, 'wb')
        p.dump(self.W, f)
        f.close()

    def load(self):
        try:
            f = open(self.fname, 'rb')
            self.W = p.load(f)
            f.close()
        except:
            print("Generating file", self.fname)
            self.load_random()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def forward(self, x):
        for i in range(len(self.W)):
            x = np.dot(x, self.W[i])
            x = self.sigmoid(x)
        return x

    def cost(self, x, y):
        return np.sum((self.forward(x)-y)**2/2, axis=1)


network = Network([1,3,3,3,1], './data/x')
network.load()

x = [[0.5],[0.6],[0.7], [0]]
y = [[xx[0]/2] for xx in x]

x += [[1]]
y += [[1]]

X = [xx[0] for xx in x]
Y = [yy[0] for yy in y]

plt.plot(X,Y, 'ro')

currC = 1

for _ in range(100):

    for k in range(len(network.W)):
        a, b = network.W[k].shape
        for l in range(a):
            for m in range(b):
                
                w = network.W[k][l][m]
                wmin = -10
                wmax = 10
                n = 8

                for i in range(10):

                    W = np.linspace(wmin, wmax, n+1)
                    C = []
                    for w in W:
                        network.W[k][l][m] = w
                        cost = sum(network.cost(x,y))
                        C += [cost]

                    index = np.argmin(C)
                    w = W[index]
                    c = C[index]

                    dw = (wmax-wmin)/n
                    wmin = w-dw
                    wmax = w+dw

                print(c)
                currC = c
                network.W[k][l][m] = w

print(network.W)

network.save()

N = 128

X = np.linspace(0,1,N)
Y = []

for x in X:
    Y += [network.forward(x)[0]]

plt.plot(X,np.array(Y))

plt.show()