import PyANNCat as pANN
import numpy as np
from random import randint
import math
import time

MNIST_TRAIN_IMG = np.fromfile('train-images-idx3-ubyte', dtype = np.uint8)
MNIST_TRAIN_IMG = MNIST_TRAIN_IMG[16:]
MNIST_TRAIN_IMG = np.reshape(MNIST_TRAIN_IMG, (60000,784))
MNIST_TRAIN_IMG = MNIST_TRAIN_IMG.T/256.0-.5
MNIST_TRAIN_LBL = np.fromfile('train-labels-idx1-ubyte', dtype = np.uint8)
MNIST_TRAIN_LBL = MNIST_TRAIN_LBL[8:]

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
xor_weights.append(np.random.randn(50,785)*(1/math.sqrt(785)))
xor_weights.append(np.random.randn(10,51)*(1/math.sqrt(51)))

errorA = np.zeros((12000))
xor_net =  pANN.FCHiddenNetwork(xor_weights, bias)
weights = np.copy(xor_weights)
for i in range(len(weights)):
		weights[i] = np.zeros(xor_net.weights[i].shape)
for j in range(0,1):
	error = 0
	weights = xor_weights
	for i in range(len(weights)):
		weights[i] = np.zeros(xor_net.weights[i].shape)
	for i in range(0, 100):
		inputs = np.column_stack((MNIST_TRAIN_IMG[:,i])).T
		trainingV = np.zeros((1,10))
		trainingV[0,MNIST_TRAIN_LBL[i]] = 1
		goal = trainingV.T
		Nerror, Nweight = xor_net.fullbackprop(inputs, goal, 0.001)
		error +=Nerror
		weights +=Nweight
	for i in range(len(weights)):
		xor_net.weights[i] += (weights[i])/100

	errorA[j] = error/100

	print errorA[j]
	np.random.shuffle(MNIST_TRAIN_IMG)
	k = 0
	for i in range(0,10000):
		inputs = np.column_stack((MNIST_TEST_IMG[:,i])).T
		if (MNIST_TEST_LBL[i] == np.argmax(xor_net.feedforward(inputs))):
			k += 1

	print k