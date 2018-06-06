import numpy as np
class MLP:
    def __init__(self, ni, nh, no):
        self.no_of_inputs = ni
        self.no_of_hidden_units = nh
        self.no_of_outputs = no
        self.weights1 = np.array
        self.weights2 = np.array
        self.delta_weights1 = np.array
        self.delta_weights2 = np.array
        self.z1 = np.array
        self.z2 = np.array
        self.h = np.array
        self.o = np.array
		
		
    ## used  when range is between 0 and one  so for xor
    def choose_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

		
		
    ## used  when range is between 0 and one  so for xor
    def choose_sigmoid_der(self, x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    ## used  when range is between -1 and 1  so for sine
    def choose_hyperbolic_tan_der(self, x):
        return 1 - (np.power(self.choose_hyperbolic_tan(x), 2))

    ## used  when range is between -1 and 1  so for sine
    def choose_hyperbolic_tan(self, x):
        return (2 / (1 + np.exp(x * -2))) - 1

    ##fill delta weights with 0 and intialise  weights to small random  values
    def randomise(self):
        self.weights1 = np.array((np.random.uniform(0.0, 1, (self.no_of_inputs, self.no_of_hidden_units))).tolist())
        self.delta_weights1 = np.dot(self.weights1, 0)
        self.weights2 = np.array((np.random.uniform(0.0, 1, (self.no_of_hidden_units, self.no_of_outputs))).tolist())
        self.delta_weights2 = np.dot(self.weights2, 0)

    ##  Computes  activations  and  use sigmoid calculationas default when sin is true  use hyperbolic tan
    def forward(self, input_vectors, sin=False):
        self.z1 = np.dot(input_vectors, self.weights1)
        if sin:
            self.h = self.choose_hyperbolic_tan(self.z1)
        else:
            self.h = self.choose_sigmoid(self.z1)

        self.z2 = np.dot(self.h, self.weights2)
        if sin:
            self.o = self.choose_hyperbolic_tan(self.z2)
        else:
            self.o = self.choose_sigmoid(self.z2)
			
			

    # Comput error  using the backprop  algorithm
    def backwards(self, inputs, target, sin=False, lan=False):
        if lan:
            val= np.sum(self.o)
            prediction_error = np.subtract(target, val)
        else:
            prediction_error = np.subtract(target, self.o)
        if sin:
            activation_out_2 = self.choose_hyperbolic_tan_der(self.z2)
            activation_out_1 = self.choose_hyperbolic_tan_der(self.z1)
        else:
            activation_out_2 = self.choose_sigmoid_der(self.z2)
            activation_out_1 = self.choose_sigmoid_der(self.z1)

        dw2_a = np.multiply(prediction_error, activation_out_2)
        self.delta_weights2 = np.dot(self.h.T, dw2_a)
        dw1_a = np.multiply(np.dot(dw2_a, self.weights2.T), activation_out_1)
        self.delta_weights1 = np.dot(inputs.T, dw1_a)
        return np.mean(np.abs(prediction_error))

    # changes weights with regard to learning rate after the deltas have been computed in backwards
    def update_weights(self, learning_rate):
        self.weights1 = np.add(self.weights1, learning_rate * self.delta_weights1)
        self.weights2 = np.add(self.weights2, learning_rate * self.delta_weights2)
        self.delta_weights1 = np.array
        self.delta_weights2 = np.array
