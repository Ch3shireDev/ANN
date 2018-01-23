from neural_network import Network
import numpy as np

x = np.array([[x] for x in np.linspace(0,1,20)])
y = np.sin((np.pi+1)*2*x)/2+1/2

network = Network([1,3,3,3,1], filename='./data/x', bias=[True])
network.load_random()

for ii in range(50):
    c = network.retarded_training(x,y)
    print(ii, c)