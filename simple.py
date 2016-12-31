import numpy as np

""" Neural Network Logic
1. take inputs from a training set, adjust by weights, pass them through formula to calculate neuron's output
2. calculate error -- difference between neuron's output and desired output in training set example
3. adjust weights based on direction of error
4. repeat process 10,000 times

Neuron's output is weighted sum of inputs
- sigmoid function used to normalize results to between 0 and 1
Output formula = 1/(1 + e^(negative weighted sum of inputs))

Weight adjustment function is proportional to error, boolean input, gradient of sigmoid function
- reason for this is that we used sigmoid curve to calculate output of the neuron
- at large numbers, sigmoid curve has a shallow gradient, meaning we don't want to adjust it too much
- gradient of sigmoid curve is just output*(1-output)
Adjustment formula = error * input * output * (1 - output)

Eventually weights reach an optimum for training set
Propogation: allowing the neuron to predict for a new situation
"""

class NeuralNetwork():
	def __init__(self):
		# choose a fixed random seed, so the rng generates the same numbers
		# every time the program runs
		np.random.seed(17)

		# model of a single neuron with 3 inputs and 1 output
		# assigning random weights to 3 x 1 matrix from -1 to 1, mean = 0
		self.synaptic_weights = 2 * np.random.random((3,1)) - 1

	def sigmoid(self, x):
		"""Sigmoid Function: pass in the weighted sum of inputs 
		for normalization between 0 to 1 in the Sigmoid curve (S shaped)"""
		return 1 / (1 + np.exp(-x))


	def sigmoid_derivative(self, x):
		"""Sigmoid Derivative: gradient of the Sigmoid curve,
		signnifies how confident we are in the existing weight"""
		return x * (1 - x)

	def train(self, training_inputs, training_outputs, num_iterations):
		""" Train the neural network by trial and error, adjusting 
		synaptic_weights each time"""
		for iteration in np.arange(num_iterations):
			# pass training set through NeuralNetwork
			output = self.calculate(training_inputs)

			# calculate error (difference between desired output 
			# and predicted output)
			error = training_outputs - output

			# multiply error by input and gradient of the Sigmoid curve
			# less confident weights adjusted more, zero inputs don't cause change to weights
			adjustment = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

			# adjust weights
			self.synaptic_weights += adjustment

	def calculate(self, inputs):
		"""Pass inputs through neural network"""
		return self.sigmoid(np.dot(inputs, self.synaptic_weights))

# Initialize single neuron neural network
neural_network = NeuralNetwork()

print("Randomly initialized synaptic_weights:")
print(neural_network.synaptic_weights)

# Training set, 4 sets of inputs/outputs
training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_outputs = np.array([[0, 1, 1, 0]]).T

# Train the neural network using training set, 10000 iterations
neural_network.train(training_inputs, training_outputs, 10000)

print("New synaptic_weights after training:")
print(neural_network.synaptic_weights)

# Test neural network with a new situation
print("Considering situation [1, 0, 0]:")
print(neural_network.calculate(np.array([1, 0, 0])))