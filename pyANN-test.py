import PyANNCat as pANN
import numpy as np
from random import randint

xor_weights = []
bias = []
bias.append(1)
bias.append(1)
xor_weights.append(np.random.random_sample((50,787)))
xor_weights.append(np.random.random_sample((10,51)))
print xor_weights[0].dtype


xor_net =  pANN.FCHiddenNetwork(xor_weights, bias)

for i in range(0, 1000):
	rand1 = randint(0, 1)
	rand0 = randint(0, 1)
	inputs = np.random.random_sample((786,1))
	goal = np.random.random_sample((10,1))
	xor_net.backprop(inputs, goal)

