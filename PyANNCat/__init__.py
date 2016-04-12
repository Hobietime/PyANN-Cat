import math
import numpy as np

#option for activation function 
TAN_H = 0

#the rate at which the neural net learns
LEARN_RATE = 0.001



def Internalerror(prevWeights, prevError, z): #pylint: disable=C0103,C0103,C0103,C0103
	"""calculates the error for the internal layers
		Args:
			costPrime (np.array): a vector that holds the change in cost for each node
			of the layer
			sigmaPZ (np.array): the derivative of the activation function
		Returns:
			the error for the last layer in a neural net
	"""
	newError = (prevWeights.T.dot(prevError) * Activateprime(z))
	return newError[0:-1, :] 

def Finalerror(costPrime, sigmaPZ):
	"""calculates the error for the final layer
		Args:
			costPrime (np.array): a vector that holds the change in cost for each node of the layer
			sigmaPZ (np.array): the derivative of the activation function
		Returns:
			the error for the last layer in a neural net
	"""
	return costPrime*sigmaPZ

def Cost(output, goal):
	"""The cost function (1/2 sum(goal - output))**2
		Args:
			output (np.array): the output of the array
			goal (np.array): what the output should be
		Returns:
			cost value
	"""
	value = np.subtract(goal, output)**2    
	return value.sum()/2

def Costprime(output, goal):
	"""The change in cost function (2*(output - goal))
		Args:
			output (np.array): the output of the array
			goal (np.array): what the output should be
		Returns:
			vector of changes in cost
	"""
	value = np.subtract(goal, output)       
	return value

def Activate(x):
	"""The default activation function
		Args:
			x (np.array): the sum of all weighted inputs into the neuron layer
		Returns:
			the activation function output
	"""
	return 1.7*(np.tanh((2.0/3) * x))
	

def Activateprime(x):
	"""The derivative of  default activation function
		Args:
			x (np.array): the sum of all weighted inputs into the neuron layer
		Returns:
			the derivative of the activation function output
		TODO: 
			make this dynamic so that I don't have to calculate a new derivative every time I change the activation function
	"""
	return (17/15) * ((np.cosh((2.0/3) * x)** -1) ** 2)



class FCHiddenNetwork:
	
	"""This class if for building ANNs with an arbitrary amount of layers, the first layer is the input layer, the output layer is the last layer
	All inner layers are hidden
	There is full connectivity, use CVNetwork for convolution networks
	"""
	def __init__(self, weights, bias):
		"""initializes the class and sets things such as starting weights and the size/shape of an the networks
			Args:
				weights (np.array[]):   a list of numpy vectors. The first array defines the weights from one v
						An example would be to move from a 2 input layer to a 4 node hidden layer, 
						layers[0] = [[a, b], 
									 [c, d],
									 [e, f],
									 [g, h]] 
			TODO: 
				add argument sigma  (int): chooses what form of activation function
		"""
		try:
			for i in range(len(weights)-1):
				if (weights[i].shape[0] + 1 != weights[i+1].shape[1]):
					raise ValueError('weights are malformed')
		except ValueError as err:
				print(err.args)
		else:
			self.weights = weights
			self.bias = bias
			self.oldError = []
			self.pweights = np.copy(self.weights)
			for i in range(len(self.pweights)):
				self.pweights[i] = np.zeros((self.pweights[i].shape))
		#too be added if different activation functions are used
		"""
		try:
			if (sigma not in sigmaList):
				raise TypeError('sigma function not defined')
		except TypeError as err:
			print(err.args)
		self.sigma = sigma
		"""

	def feedforward(self, inputs):
		"""runs a set of inputs through the neural net
			Args:
				inputs (np.array): 1 dimensional array of inputs

			Returns:
				1 dimensional array of outputs
		"""
		
		inputs = np.append(inputs, [[self.bias[0]]], axis = 0)
		for i in range(0, len(self.weights) - 1):
			inputs = np.dot(self.weights[i], inputs)
			inputs = np.append(inputs, [[self.bias[i+1]]], axis = 0)
			inputs = Activate(inputs) 
		inputs = np.dot(self.weights[len(self.weights) - 1], inputs)
		inputs = Activate(inputs) 
		return inputs
		

	#@profile
	def feedforwardfb(self, inputs):
		"""runs a set of inputs through the neural net
			Args:
				inputs (np.array): 1 dimensional array of inputs

			Returns:
				a list of arrays of intermediate summations and outputs
		"""
		z = []
		a = []
		inputs = np.append(inputs, [[self.bias[0]]], axis = 0)
		a.append(inputs)
		for i in range(0, len(self.weights) - 1):
			inputs = np.dot(self.weights[i], inputs)
			inputs = np.append(inputs, [[self.bias[i+1]]], axis = 0)
			z.append(inputs)
			inputs = Activate(inputs) 
			a.append(inputs)
		inputs = np.dot(self.weights[len(self.weights) - 1], inputs)
		z.append(inputs)
		inputs = Activate(inputs) 
		a.append(inputs)
		return a, z

	#@profile
	def backprop(self, inputs, goal):
		"""runs a set of inputs through the neural net, then preforms back propagation based on the goal provided
			Args:
				inputs (np.array): 1 dimensional array of inputs
				goal (np.array): 1 dimensional array of desired outputs

			Returns:
				current error value (as a np.array)
		"""
		a, z = self.feedforwardfb(inputs)
		costGrad = Costprime(a[-1], goal)
		cost = Cost(a[-1], goal)
		outprime = Activateprime(z[-1])
		error = Finalerror(costGrad, outprime)
		for i in range(len(self.weights) - 1, 0, -1):
			self.weights[i] += (LEARN_RATE*cost**2) * (a[i].T * (error))
			error = Internalerror(self.weights[i], error, z[i-1])
		self.weights[0] += (a[0].T * error).dot(LEARN_RATE *cost**2)
		return cost

	#@profile
	def backprop(self, inputs, goal, learn_rate):
		"""runs a set of inputs through the neural net, then preforms back propagation based on the goal provided
			Args:
				inputs (np.array): 1 dimensional array of inputs
				goal (np.array): 1 dimensional array of desired outputs
				learn_rate (float): a float thatt defines the learn rate

			Returns:
				current error value (as a np.array)
		"""
		a, z = self.feedforwardfb(inputs)
		costGrad = Costprime(a[-1], goal)
		cost = Cost(a[-1], goal)
		outprime = Activateprime(z[-1])
		error = Finalerror(costGrad, outprime)
		for i in range(len(self.weights) - 1, 0, -1):
			self.weights[i] += (learn_rate*cost**2) * (a[i].T * (error))
			error = Internalerror(self.weights[i], error, z[i-1])
		self.weights[0] += (a[0].T * error).dot(learn_rate *cost**2)
		return cost

	def Momentumbackprop(self, inputs, goal, learn_rate, momentum):
		"""runs a set of inputs through the neural net, then preforms back propagation based on the goal provided
			Args:
				inputs (np.array): 1 dimensional array of inputs
				goal (np.array): 1 dimensional array of desired outputs
				learn_rate (float): a float that defines the learn rate

			Returns:
				the calculated differences in the weights
		"""
		a, z = self.feedforwardfb(inputs)
		costGrad = Costprime(a[-1], goal)
		cost = Cost(a[-1], goal)
		outprime = Activateprime(z[-1])
		error = Finalerror(costGrad, outprime)
		for i in range(len(self.weights) - 1, 0, -1):
			self.weights[i] += (learn_rate*cost**2) * (a[i].T * (error)) + (self.pweights[i] * momentum)
			self.pweights[i] = (learn_rate*cost**2) * (a[i].T * (error))
			error = Internalerror(self.weights[i], error, z[i-1])
		self.weights[0] += (a[0].T * error).dot(learn_rate *cost**2) + (self.pweights[0] * momentum)
		self.pweights[0] = (a[0].T * error).dot(learn_rate *cost**2)

		return cost

	def miniBatchprop(self, inputs, trainAr, batchSize, learn_rate):
		"""TO DO
		"""
		aC = 0
		zC = 0
		cost = 0
		weights = np.copy(self.weights)
		for i in range(len(weights)):
			weights[i] = np.zeros(weights[i].shape)
		for i in range(batchSize):
			trainingV = np.zeros((1,10))
			trainingV[0,trainAr[i]] = 1
			goal = trainingV.T
			a, z = self.feedforwardfb(inputs[0:784,i].reshape((784, 1)))
			costGrad = Costprime(a[-1], goal)
			cost += Cost(a[-1], goal)
			outprime = Activateprime(z[-1])
			error = Finalerror(costGrad, outprime)
			for k in range(len(weights) - 1, 0, -1):
				weights[k] += (learn_rate) * (a[k].T * (error))
				error = Internalerror(self.weights[k], error, z[k-1])
			weights[0] += (a[0].T * error).dot(learn_rate)

		for j in range(len(weights)):
			self.weights[j] += ((weights[j] + self.pweights[j]*.1) /batchSize) 
		self.pweights = np.copy(weights)
		cost /= float(batchSize)
		return cost
			
