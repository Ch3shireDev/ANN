import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def plot(forward, X, Y, err):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    hoursSleep = np.linspace(0, 10, 100)
    hoursStudy = np.linspace(0, 5, 100)
    hoursSleepNorm = hoursSleep/10.
    hoursStudyNorm = hoursStudy/5.
    a, b  = np.meshgrid(hoursSleepNorm, hoursStudyNorm)
    allInputs = np.zeros((a.size, 2))
    allInputs[:, 0] = a.ravel()
    allInputs[:, 1] = b.ravel()
    allOutputs = forward(allInputs)
    yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
    xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T
    ax.scatter(10*X[:,0], 5*X[:,1], 100*Y, c='k', alpha = 1, s=30)
    surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100, 100), cmap=cm.jet, alpha = 0.5)
    ax.set_xlabel('Hours Sleep')
    ax.set_ylabel('Hours Study')
    ax.set_zlabel('Test Score')
    plt.suptitle("error value:" + err)
    ax.view_init(10,-150)
    plt.savefig(err+'.png')
    print("Saved plot to", err+".png")