import numpy as np
import pickle as p

def sigmoid(x):
    return 1/(1+np.exp(-x))

def error(x, y):
    return np.sum((x-y)**2)/2

fname = "brain.dat"

Nin = 2
Nh = 3
Nout = 1


fname = "brain"+('-'.join([str(s) for s in [Nin,Nh,Nout]]))+".dat"

W1, W2 = None, None


try:
    f = open(fname, 'rb')
    W1, W2 = p.load(f)
    f.close()
    if W1.shape != (Nin,Nh) or W2.shape != (Nh,Nout):
        raise Exception
except:
    print("Error:", fname, "corrupted or missing, creating new synapsis data...")
    W1 = np.random.randn(Nin, Nh)
    W2 = np.random.randn(Nh, Nout)


X = np.array(([3,5], [5,1], [10,2]), dtype=float)
Y = np.array(([75], [82], [93]), dtype=float)

X = X/np.amax(X, axis=0)
Y = Y/100 #Max test score is 100

def forward(X):
    x1 = np.dot(X, W1)
    x1 = sigmoid(x1)
    x2 = np.dot(x1, W2)
    x2 = sigmoid(x2)
    return x2

print(error(Y, forward(X)))


# save progress

f = open(fname, 'wb')
p.dump((W1, W2), f)
f.close()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

fig = plt.figure()
ax = fig.gca(projection='3d')

#Test network for various combinations of sleep/study:
hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)

#Normalize data (same way training data way normalized)
hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.

#Create 2-d versions of input for plotting
a, b  = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

#Join into a single input matrix:
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()

allOutputs = forward(allInputs)

yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

ax.scatter(10*X[:,0], 5*X[:,1], 100*y, c='k', alpha = 1, s=30)

surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100), \
                       cmap=cm.jet, alpha = 0.5)


ax.set_xlabel('Hours Sleep')
ax.set_ylabel('Hours Study')
ax.set_zlabel('Test Score')