import png
import numpy as np


class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def forward(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))


a = Neural_Network()

f = open('example.png', 'rb')      # binary mode is important
reader = png.Reader(file=f)
x, y, pixels, meta = reader.asDirect()
tab = list(pixels)
f.close()

tab = [[0 if x>128 else 1 for x in t] for t in tab]

h0, w0 = len(tab), len(tab[0])

h = h0//10
w = w0//20

print(np.array(tab).shape)

print(w)

tab = [[[[tab[y][x] for x in range(i*w,(i+1)*w)] for y in range(j*h,(j+1)*h)] for i in range(20)] for j in range(10)]

for i in range(10):
    for x in tab[i][2]:
        for j in x:
            print(j,end="")
        print()


# n, m = len(x),len(x[0])

# for y in x:
#     print(len(''.join([str(a) for a in y])))

