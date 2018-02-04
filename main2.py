import numpy as np
import pickle as p
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn
from neural import network

nx = 2
ny = 2

net = network([nx,8,8,ny], filename='./data/aaa', bias=True)

x = randn(nx)
y0 = net.forward(x)[0]

net.expand(1)

y1 = net.forward(x)[0]

dy = sum(y1-y0)

print(dy)