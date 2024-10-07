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
        # Apply the ReLU activation function
        self.output = relu(self.output)
        return self.output


class Layer:

    def init(self, num_inputs, num_neurons, activation_function=relu):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        # Creating the neurons
        # Here, create the number of neurons required and store them in a list
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.activation_function = activation_function
        self.outputs = []

    def forward(self, inputs):
        # Take the inputs and pass them each neuron's forward functions and store the outputs in the self.outputs list
        self.outputs = [neuron.forward(inputs) for neuron in self.neurons]


class NeuralNetwork:

    def init(self, num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_outputs = num_outputs

        # Now that we have all the required variables, go ahead and create the layers

        # Always remember that we do NOT need to create a layer for the inputs. The initial
        # inputs that we get make up the first input layer. So, we start from the first hidden
        # layer and create layers all the way up to the last (output) layer

        self.layers = []

        # Create the hidden layers
        for _ in range(num_hidden_layers):
            self.layers.append(Layer(num_inputs, num_hidden_layer_neurons))
            num_inputs = num_hidden_layer_neurons  # Update num_inputs for the next layer

        # Create the output layer
        self.layers.append(Layer(num_hidden_layer_neurons, num_outputs))

        # Create the appropriate number of hidden layers each with the appropriate number of neurons
        # Tip: Use a for loop to create the hidden layers

        # At the end, create the output layer
        output_layer = Layer(num_hidden_layer_neurons, num_outputs)


    def forward(self, inputs):
        '''
        Take the inputs and pass those inputs to each layer in the network
        Tip, use a for loop and one variable to keep track of the outputs of a single layer
        Keep updating that single variable with the outputs of the layers
        At the end, whatever is in that variable will be the output of the last layer
        '''

        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.outputs

        return inputs