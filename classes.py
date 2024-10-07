import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
    return exp_x / exp_x.sum(axis=0)

class Neuron:
    def __init__(self, num_inputs, activation_function=relu):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.activation_function = activation_function
        self.output = 0

    def forward(self, inputs):
        # Calculate the dot product and add the bias
        self.output = np.dot(self.weights, inputs) + self.bias
        # Apply the activation function
        self.output = self.activation_function(self.output)
        return self.output

class Layer:
    def __init__(self, num_inputs, num_neurons, activation_function=relu):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        # Creating the neurons
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]
        self.activation_function = activation_function
        self.outputs = []

    def forward(self, inputs):
        # Take the inputs and pass them to each neuron's forward function and store the outputs in the self.outputs list
        self.outputs = [neuron.forward(inputs) for neuron in self.neurons]
        # Apply the activation function to the entire layer's outputs if it's softmax
        self.outputs = self.activation_function(np.array(self.outputs))
        self.outputs = softmax(self.outputs)
        return self.outputs

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_outputs = num_outputs

        # Create the layers
        self.layers = []

        # Create the hidden layers with relu activation function
        for _ in range(num_hidden_layers):
            self.layers.append(Layer(num_inputs, num_hidden_layer_neurons, activation_function=relu))
            num_inputs = num_hidden_layer_neurons  # Update num_inputs for the next layer

        # Create the output layer with softmax activation function
        self.layers.append(Layer(num_hidden_layer_neurons, num_outputs, activation_function=softmax))

    def forward(self, inputs):
        '''
        Take the inputs and pass those inputs to each layer in the network
        '''
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.outputs

        return inputs