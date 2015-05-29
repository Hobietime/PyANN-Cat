import PyANNCat as pANN
import numpy as np
from random import randint

xor_weights = []
bias = []
bias.append(1)
bias.append(1)
xor_weights.append(np.random.randn(6,3))
xor_weights.append(np.random.randn(1,7))
print xor_weights[0].dtype


xor_net =  pANN.FCHiddenNetwork(xor_weights, bias)

for i in range(0, 10000):
	rand1 = randint(0, 1)
	rand0 = randint(0, 1)
	inputs = np.array([[rand0], [rand1]])
	print inputs
	goal = rand1 ^ rand0
	print goal
	print xor_net.backprop(inputs, goal)

print xor_net.feedforward([[0],[1]])
print xor_net.feedforward([[1],[1]])
print xor_net.feedforward([[0],[0]])
print xor_net.feedforward([[1],[0]])