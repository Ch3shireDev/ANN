import numpy as np
import pickle as p
from itertools import product
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn, randint
from copy import deepcopy

class network:
    
    def __init__(self, N=[1,3,1], bias=None, filename = "unnamed"):
        if(len(N)<2):
            print("Wrong size of N. len(N) should be greater than 1.")
            exit()
        self.N = N
        N = len(N)
        self.fname = None
        self.Bias = [False for _ in range(1, N)]
        self.Wzero = None
        self.indices = None
        if bias == True:
            self.Bias = [True for _ in range(1, N)]
        if type(bias) is list:
            for i in range(len(bias)):
                self.Bias[i] = bias[i]
        self.filename = filename

        self.load_random()

    def expand(self, n):
        # self.N[n] += 1
        N = self.N

        Wa = self.W[n]
        Wb = self.W[n+1]

        na, nb = N[n-1]+(1 if self.Bias[n-1] else 0), N[n]
        # nc, nd = N[n]+(1 if self.Bias[n] else 0), N[n+1]

        self.W[n] = np.array([[Wa[j][i] for i in range(na)] for j in range(nb)])
        # self.W[n+1] = np.array([[Wb[i][j] for i in range(nd)] for j in range(nc)])


    def load_random(self):
        N = self.N
        self.W = np.array([randn(N[i-1]+(1 if self.Bias[i-1] else 0), N[i]) for i in range(1,len(N))])

    def getFilename(self):
        fname = self.filename + ('-'.join([str(s) for s in self.N]))
        if np.any(self.Bias):
            fname += "-B"
            for b in self.Bias:
                fname += '1' if b else '0'
        fname += ".dat"
        return fname

    def save(self):        
        f = open(self.getFilename(), 'wb')
        p.dump(self.W, f)
        f.close()

    def load(self):
        fname = self.getFilename()
        try:
            f = open(fname, 'rb')
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
        self.z = []
        self.a = []
        for i in range(len(W)):
            WW = W[i]
            if self.Bias[i]==True:
                y = np.array([[1] for _ in x])
                x = np.concatenate([y, np.array(x)], axis=1)
            x = np.dot(x, WW)
            self.z += [x]
            x = sigmoid(x)
            self.a += [x]
        return x

    def cost(self, x, y, W=None):
        return np.sum((self.forward(x, W)-y)**2/2, axis=1)

    def retarded_training(self, x, y, Wmin=-10, Wmax=10, n=8):
        
        Wcopy = deepcopy(self.W)
        c_begin = sum(self.cost(x,y))

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

        if c > c_begin:
            self.W = Wcopy
        return c

    ### kills one synapse. It's weird but helps

    def shake_internal(self):    
        a = len(self.W)
        a = randint(0,a)
        b, c = self.W[a].shape
        b = randint(0, b)
        c = randint(0, c)
        self.W[a][b][c]=randn()

    def shake(self,x,y,n=20,m=5):
        W = deepcopy(self.W)
        c0 = sum(self.cost(x,y))

        for i in range(m):
            self.shake_internal()
            for j in range(n):
                self.retarded_training(x,y)
            c1 = sum(self.cost(x,y))
            if c1<c0:
                print("Success!")
                break
            else:
                self.W = deepcopy(W)
                print(i, "Failure")


    def gradient_training(self, x, y, Wmin=-10, Wmax=10, n=8, dw=0.0001):
        
        if self.Wzero is None:
            self.Wzero = np.array([np.zeros_like(w) for w in self.W])
        
        if self.indices is None:
            indices = []
            for k in range(len(self.W)):
                a, b = self.W[k].shape
            for l, m in product(range(a), range(b)):
                indices += [(k,l,m)]
            self.indices = indices

        indices = self.indices
        Wzero = self.Wzero
        W = self.W
        
        y0 = sum(self.cost(x,y))
        dW = np.array([w.copy() for w in Wzero])

        #from backpropagation algorithm
        # yHat = self.forward(X)
        # delta3 = np.multiply(-(y-yHat), sigmoidPrime(z3))
        # dJdW2 = np.dot(a2.T, delta3)
        # delta2 = np.dot(delta3, W2.T)*sigmoidPrime(z2)
        # dJdW1 = np.dot(X.T, delta2)  

        yHat = self.forward(x)
        
        delta = np.multiply(yHat - y, sigmoidPrime(self.z[-1]))
        dJdW  = np.dot(self.a[-2].T, delta)

        for k, l, m in indices:
            Wzero[k][l][m] = dw
            y1 = sum(self.cost(x,y,W+Wzero))
            dW[k][l][m] = (y1-y0)/dw
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