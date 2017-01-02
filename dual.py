import numpy as np

""" 
Vinitra Swamy
1/1/17

	Multilayer NeuralNetwork
	- often, a single neuron is not adept to handling more complicated cases
	- namely, nonlinear relationships -- if there is not a direct one-to-one relationship between inputs and outputs

	Solution: create an additional hidden layer!
	- in this case, we'll create a layer of four neurons that enables neural net to think about combinations of inputs
	- adding more hidden layers is a process called deep learning

	Very useful for image recognition:
	- no direct relationship between pixels and bananas, but direct relation between combinations of pixels and bananas
"""

class NeuronLayer():
	def __init__(self, num_neurons, num_inputs):
		# num_inputs corresponds to the number of inputs for each neuron
		self.synaptic_weights = 2 * np.random.random((num_inputs, num_neurons)) - 1

class NeuralNetwork():
	def __init__(self, layer1, layer2):
		self.layer1 = layer1
		self.layer2 = layer2

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
			# pass training set through NeuralNetwork to get layer 1 and 2 outputs
			output_l1, output_l2 = self.calculate(training_inputs)

			# calculate error for layer 2 (difference between desired output 
			# and predicted output)
			l2_error = training_outputs - output_l2
			l2_delta = l2_error * self.sigmoid_derivative(output_l2)

			# calculate error for layer 1 (by looking at layer 1 
			# weights, we can find how much layer 1 contributed to layer 2 error)
			l1_error = l2_delta.dot(self.layer2.synaptic_weights.T)  # dot product of weights from transformed layer 2
			l1_delta = l1_error * self.sigmoid_derivative(output_l1)

			# multiply error by input and gradient of the Sigmoid curve
			# less confident weights adjusted more, zero inputs don't cause change to weights
			l1_adjustment = training_inputs.T.dot(l1_delta)
			l2_adjustment = output_l1.T.dot(l2_delta)

			# adjust weights for multiple layers
			self.layer1.synaptic_weights += l1_adjustment
			self.layer2.synaptic_weights += l2_adjustment

	def calculate(self, inputs):
		"""Pass inputs through neural network"""
		output_l1 = self.sigmoid(np.dot(inputs, self.layer1.synaptic_weights))
		output_l2 = self.sigmoid(np.dot(output_l1, self.layer2.synaptic_weights))
		return output_l1, output_l2

	def print_weights(self):
		print("Layer 1 (4 neurons, 3 inputs each): ")
		print(self.layer1.synaptic_weights)
		print("Layer 2 (2 neuron, 4 inputs): ")
		print(self.layer2.synaptic_weights)

#Seed the random number generator
np.random.seed(1)

# Create layer 1 (4 neurons, each with 3 inputs)
layer1 = NeuronLayer(4, 3)

# Create layer 2 (2 neurons, each with 4 inputs)
layer2 = NeuronLayer(2, 4)

# Combine the layers to create a neural network
neural_network = NeuralNetwork(layer1, layer2)

print("Stage 1) Random starting synaptic weights: ")
neural_network.print_weights()

# The training set. We have 7 examples, each consisting of 3 input values
# and 1 output value.
training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

# Train the neural network using the training set.
# Do it 60,000 times and make small adjustments each time.
neural_network.train(training_set_inputs, training_set_outputs, 60000)

print("Stage 2) New synaptic weights after training: ")
neural_network.print_weights()

# Test the neural network with a new situation.
print("Stage 3) Predicting a new situation [1, 1, 0]: ")
hidden_state, output = neural_network.calculate(np.array([1, 1, 0]))
print(output)