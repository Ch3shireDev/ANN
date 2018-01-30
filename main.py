import numpy as np
import pickle as p
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.random import randn
from neural_network import network

net = network([1,8,8,1], filename='./data/x')
net.load_random()
# net.load()

n = 8

x = [[x] for x in np.linspace(0,1,n)]
y = [[np.random.random()] for _ in range(n)]
# y = [[0.4476942599988286], [0.7627471086491007], [0.2820267523688108], [0.25279662378888723]]
# y =  [[0.2502694722847021], [0.3462503496426532], [0.8970113465326002], [0.2842288590696336]]
# y = [[0.7051114931826049], [0.13568097772897503], [0.6205768631833326], [0.42645564603793074]]
# y = [[0.07596822512991885], [0.5016563812593563], [0.6612149390864971], [0.6805438125041167]]

X = [xx[0] for xx in x]
Y = [yy[0] for yy in y]



c = 1
ii=0

while c>0.001:
# for ii in range(100):
    # c = net.gradient_training(x,y,dw=0.00001)
    
    c = net.retarded_training(x,y)
    if ii>0 and ii%100==0:
        net.shake()
    print(ii,c,y)
    ii+=1

N = 128

plt.plot(X,Y, 'ro')

X = np.linspace(0,1,N)
Y = []

for x in X:
    Y += [net.forward(x)[0]]

plt.plot(X,np.array(Y))
plt.show()


 