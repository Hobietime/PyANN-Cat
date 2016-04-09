import PyANNCat as pANN
import numpy as np
from random import randint
import math
import time
from tempfile import TemporaryFile

def go():
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

	MNIST_TRAIN_SET = np.zeros((785,60000))

	MNIST_TRAIN_SET[0:784, :] = MNIST_TRAIN_IMG
	MNIST_TRAIN_SET[784,:] = MNIST_TRAIN_LBL
	"""print MNIST_TRAIN_SET[:,5]
	np.random.shuffle(MNIST_TRAIN_SET.T)
	print MNIST_TRAIN_SET[:,5]
	exit()"""
	xor_weights = []
	bias = []
	bias.append(1)
	bias.append(1)
	xor_weights.append(np.random.randn(100,785)*(1/math.sqrt(785)))
	xor_weights.append(np.random.randn(10,101)*(1/math.sqrt(101)))


	errorA = np.zeros((10000))
	k = np.zeros((10000))
	xor_net =  pANN.FCHiddenNetwork(xor_weights, bias)
	for j in range(0,10000):
		for i in range(0, 60000):
			inputs = np.column_stack((MNIST_TRAIN_SET[0:784,i])).T
			trainingV = np.zeros((1,10))
			trainingV[0,MNIST_TRAIN_SET[784,i]] = 1
			goal = trainingV.T
			error = xor_net.backprop(inputs, goal, 0.00014)
		
		errorA[j] = error
		print errorA[j]
		np.random.shuffle(MNIST_TRAIN_SET.T)
		k[j] = 0
		for i in range(0,10000):
			inputs = np.column_stack((MNIST_TEST_IMG[:,i])).T
			if (MNIST_TEST_LBL[i] == np.argmax(xor_net.feedforward(inputs))):
				k[j] += 1

		print k[j]

	np.savetxt('testfile-2', k, delimiter=',')
	np.savetxt('errorfile-2', errorA, delimiter=',')

if (__name__ == "__main__"):
	go()
