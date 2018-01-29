import numpy as np
import pickle as p
from itertools import product
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn
from copy import deepcopy

class network:
    
    def __init__(self, N=[1,3,1], bias=None, filename = "unnamed"):
        if(len(N)<2):
            print("Wrong size of N. len(N) should be greater than 1.")
            exit()
        self.N = N
        self.fname = None
        self.Bias = [False for _ in range(1, len(N))]
        if bias == True:
            self.Bias = [True for _ in range(1, len(N))]
        if type(bias) is list:
            for i in range(len(bias)):
                self.Bias[i] = bias[i]
        self.fname = filename + ('-'.join([str(s) for s in N]))
        print(self.Bias, np.any(self.Bias))
        if np.any(self.Bias):
            self.fname += "-B"
            for b in self.Bias:
                self.fname += '1' if b else '0'
        self.fname += ".dat"
        print(self.fname)

    def load_random(self):
        N = self.N
        self.W = np.array([randn(N[i-1]+(1 if self.Bias[i-1] else 0), N[i]) for i in range(1,len(N))])

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

    def forward(self, x, W=None):
        x = np.array(x)
        if W is None:
            W = self.W
        if len(x.shape)==1:
            x = np.array([[xx for xx in x]])
        for i in range(len(W)):
            WW = W[i]
            if self.Bias[i]==True:
                y = np.array([[1] for _ in x])
                x = np.concatenate([y, np.array(x)], axis=1)
            x = np.dot(x, WW)
            x = sigmoid(x)
        return x

    def cost(self, x, y, W=None):
        return np.sum((self.forward(x, W)-y)**2/2, axis=1)

    def retarded_training(self, x, y, Wmin=-10, Wmax=10, n=8):
        for k in range(len(self.W)):
            a, b = self.W[k].shape
            for l, m in product(range(a), range(b)):
                w = self.W[k][l][m]
                wmin = Wmin
                wmax = Wmax
                for i in range(10):
                    W = np.linspace(wmin, wmax, n+1)
                    C = []
                    for w in W:
                        self.W[k][l][m] = w
                        cost = sum(self.cost(x,y))
                        C += [cost]
                    index = np.argmin(C)
                    w = W[index]
                    c = C[index]
                    dw = (wmax - wmin)/n
                    wmin, wmax = w - dw, w + dw
                self.W[k][l][m] = w
        return c

    def gradient_training(self, x, y, Wmin=-10, Wmax=10, n=8, dw=0.0001):
        Wzero = self.W.copy()

        for i in range(len(Wzero)):
            Wzero[i] = np.zeros_like(Wzero[i])

        indices = []
        
        W = self.W
        
        for k in range(len(Wzero)):
            a, b = Wzero[k].shape
            for l, m in product(range(a), range(b)):
                indices += [(k,l,m)]
        
        y0 = sum(self.cost(x,y,W))
        dW = np.array([w.copy() for w in Wzero])

        #from backpropagation algorithm
        # yHat = self.forward(X)
        # delta3 = np.multiply(-(y-yHat), sigmoidPrime(z3))
        # dJdW2 = np.dot(a2.T, delta3)
        # delta2 = np.dot(delta3, W2.T)*sigmoidPrime(z2)
        # dJdW1 = np.dot(X.T, delta2)  

        for k, l, m in indices:
            Wzero[k][l][m] = dw
            y1 = sum(self.cost(x,y,W+Wzero))
            dW[k][l][m] = (y1-y0)
            Wzero[k][l][m] = 0

        dW2 = deepcopy(Wzero)

        for i in range(len(dW)):
            w = np.linalg.norm(dW[i])
            if w>1e-15:
                dW[i]/=w
            wmin = Wmin
            wmax = Wmax
            for _ in range(10):
                Ws = np.linspace(wmin, wmax, n+1)
                C = []
                for w in Ws:
                    dW2[i] = dW[i]*w
                    costs = self.cost(x,y,self.W+dW2)
                    C += [sum(costs)]
                index = np.argmin(C)
                w = Ws[index]
                c = C[index]
                dw = (wmax - wmin)/n
                wmin, wmax = w - dw, w + dw
        W += dW2
        return c

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)