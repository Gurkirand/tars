import random
import numpy as np
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
    
class Network:
    def __init__(self, sizes):
        #First layer is input and last layer is output
        self.num_layers = len(sizes)
        self.sizes = sizes
        np.random.seed(0)
        
        #Returns random matrix of size (y, 1) with normal distribution of values
        #size[1:] because no bias is needed for the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        
        #Returns random matrix of size (y, x) with normal distribution of values
        #constructs weights connecting two layers
        #x from input to second to last layer
        #y from second to last layer
        #size(y, x) since our weighted input is w * x
        self.weights = [np.random.randn(y, x) 
                                for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
            
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        
        for j in xrange(epochs):
            #randomize our training set so that our batches will be random
            random.shuffle(training_data)
            
            mini_batches = [training_data[k:k + mini_batch_size]
                                for k in xrange(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
                
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                                j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
                
    def update_mini_batch(self, mini_batch, learning_rate):
        #gradients for b and w
        #initialized to zero
        full_x = mini_batch[0][0]
        full_y = mini_batch[0][1]
        
        for i in range(1, len(mini_batch)):
            full_x = np.concatenate((full_x, mini_batch[i][0]), axis=1)
            full_y = np.concatenate((full_y, mini_batch[i][1]), axis=1)
            
        nabla_b, nabla_w = self.backprop(full_x, full_y)
            
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                                for w, nw in zip(self.weights, nabla_w)]
        
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                                for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        #track activations for backprop
        activations = [x]
        #weighted inputs
        zs = []
        
        #forward pass will tracking our weighted inputs and activations
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        #calculate sigmoid prime since our backpropagation formula for error
        #in layer L (last layer) is:
        # (dC / da L) * (da L / dz L) or
        # (dC / da L) * sigmoid_prime(dz L)
        sp = sigmoid_prime(zs[-1])
        delta = self.cost_derivative(activations[-1], y) * sp
        
        #formula for change in bias in layer l is:
        #delta l
        nabla_b[-1] = delta.sum(axis=1).reshape((delta.shape[0], 1))
        
        #formula for change in weight in layer l is:
        #delta l * (a l-1)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in xrange(2, self.num_layers):
            
            #formula for layer l based on layer l+1 is:
            # ((W l+1)T * delta l+1) * sigmoid_prime(z l)
            z = zs[-l]
            sp = sigmoid_prime(z)
            
            #delta tracks our changes from layer to layer
            delta = np.dot(self.weights[-1 + l].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(axis=1).reshape((delta.shape[0], 1))
            nabla_w[-l] = np.dot(delta, activations[-1 - l].transpose())
        
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                                for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
