import PyANNCat as pANN
import numpy as np
from random import randint
import math

MNIST_TRAIN_IMG = np.fromfile('train-images-idx3-ubyte', dtype = np.uint8)
MNIST_TRAIN_IMG = MNIST_TRAIN_IMG[16:]
MNIST_TRAIN_IMG = np.reshape(MNIST_TRAIN_IMG, (60000,784))
MNIST_TRAIN_IMG = MNIST_TRAIN_IMG.T/2566.0 - .5
MNIST_TRAIN_LBL = np.fromfile('train-labels-idx1-ubyte', dtype = np.uint8)
MNIST_TRAIN_LBL = MNIST_TRAIN_LBL[8:]

MNIST_TEST_LBL = np.fromfile('t10k-labels-idx1-ubyte', dtype = np.uint8)
MNIST_TEST_LBL = MNIST_TEST_LBL[8:]
MNIST_TEST_IMG = np.fromfile('t10k-images-idx3-ubyte', dtype = np.uint8)
MNIST_TEST_IMG = MNIST_TEST_IMG[16:]
MNIST_TEST_IMG = np.reshape(MNIST_TEST_IMG, (10000,784))
MNIST_TEST_IMG = MNIST_TEST_IMG.T/256.0 - .5

MNIST_TRAIN_SET = np.zeros((785,60000))

MNIST_TRAIN_SET[0:784, :] = MNIST_TRAIN_IMG
MNIST_TRAIN_SET[784,:] = MNIST_TRAIN_LBL


xor_weights = []
bias = []
bias.append(1)
bias.append(1)
xor_weights.append(np.random.randn(50,785)*(1/math.sqrt(785)))
xor_weights.append(np.random.randn(10,51)*(1/math.sqrt(101)))
trainingV = np.zeros((10, 60000))


for k in MNIST_TRAIN_LBL:
		trainingV[MNIST_TRAIN_LBL[k], k] = 1


errorA = np.zeros((2400000))
k = np.zeros((400))

xor_net =  pANN.FCHiddenNetwork(xor_weights, bias)
for j in range(0,10):
	for i in range(0, 60000, 100):
		inputs = np.column_stack((MNIST_TRAIN_IMG[0:784,i:i+100])).T
		trainingLBLAr = MNIST_TRAIN_SET[784,i:i+100]
		

		errorA[i] = xor_net.miniBatchprop(inputs, trainingLBLAr, 100, .01)
		print errorA[i]
		#print xor_net.weights
	#np.random.shuffle(MNIST_TRAIN_SET.T)
	for i in range(0,10000):
		inputs = np.column_stack((MNIST_TEST_IMG[:,i])).T
		if (MNIST_TEST_LBL[i] == np.argmax(xor_net.feedforward(inputs))):
			k[j] += 1

	np.savetxt('minibatch-testfile', k, delimiter=',')
	np.savetxt('minbatch-errorfile', errorA, delimiter=',')