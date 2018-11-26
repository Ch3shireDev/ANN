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


network = Network([2,3,3,3,1], './data/x')
network.load()

x = [[1,1], [0.5,0.5], [0.5,0.4]]
y = [[1],[0.5],[0.7]]


xx,yy,zz = [],[],[]

for i in range(len(x)):
    xx += [x[i][0]]
    yy += [x[i][1]]
    zz += y[i]

print("xx")

currC = 1

for _ in range(100):

    for k in range(len(network.W)):
        a, b = network.W[k].shape
        for l in range(a):
            for m in range(b):
                
                w = network.W[k][l][m]
                wmin = -5
                wmax = 5
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


network.save()

N = 128

X = np.linspace(0,1,N)
Y = []

YY = []

for x in X:
    for y in X:
        z = network.forward([x,y])
        Y += [z[0]]


Y = np.array(Y)
Y = Y.reshape(N,N)
Z = np.copy(Y)

X, Y = np.meshgrid(X,X)


fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(X,Y,Z.T)


ax.plot(xx,yy,zz, 'ro')
ax.set_title(r'Artificial Neural Network (2 inputs, 1 output, 3 hidden layers)', fontsize=12)

ax.set_xlabel('Input X', color='blue')
ax.set_ylabel('Input Y', color='blue')
ax.set_zlabel('Output Z', color='blue')

plt.show()
