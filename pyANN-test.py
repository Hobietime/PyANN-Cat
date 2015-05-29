import PyANNCat as pANN
import numpy as np
from random import randint

MNIST_TRAIN_IMG = np.fromfile('train-images-idx3-ubyte', dtype = np.uint8)
MNIST_TRAIN_IMG = MNIST_TRAIN_IMG[16:]
MNIST_TRAIN_IMG = np.reshape(MNIST_TRAIN_IMG, (60000,784))
MNIST_TRAIN_IMG = MNIST_TRAIN_IMG.T/256.0-.5
MNIST_TRAIN_LBL = np.fromfile('train-labels-idx1-ubyte', dtype = np.uint8)
MNIST_TRAIN_LBL = MNIST_TRAIN_LBL[8:]
print MNIST_TRAIN_IMG[:,1]

MNIST_TEST_LBL = np.fromfile('t10k-labels-idx1-ubyte', dtype = np.uint8)
MNIST_TEST_LBL = MNIST_TEST_LBL[8:]
MNIST_TEST_IMG = np.fromfile('t10k-images-idx3-ubyte', dtype = np.uint8)
MNIST_TEST_IMG = MNIST_TEST_IMG[16:]
MNIST_TEST_IMG = np.reshape(MNIST_TEST_IMG, (10000,784))
MNIST_TEST_IMG = MNIST_TEST_IMG.T/256.0-.5

xor_weights = []
bias = []
bias.append(1)
bias.append(1)
xor_weights.append(np.random.randn(4,785)*.02)
xor_weights.append(np.random.randn(10,5)*.02)


xor_net =  pANN.FCHiddenNetwork(xor_weights, bias)

for i in range(0, 6000):
	inputs = np.column_stack((MNIST_TRAIN_IMG[:,i])).T
	trainingV = np.zeros((1,10))
	trainingV[0,MNIST_TRAIN_LBL[i]] = 1
	goal = trainingV.T
	#print inputs
	#print goalq
	print xor_net.backprop(inputs, goal)
	#print xor_net.weights

for i in range(0,2):
	inputs = np.column_stack((MNIST_TEST_IMG[:,i])).T
	print MNIST_TEST_LBL[i]
	print xor_net.feedforward(inputs)