import numpy as np;
import tensorflow as tf;

class RNN_cell:
    def __init__(self, i_size, h_size, o_size):
        self.input_size = i_size
        self.hidden_size = h_size
        self.output_size = o_size
    
        self.Wx = self.weight_init([self.input_size, self.hidden_size])
        self.Wh = self.weight_init([self.hidden_size, self.hidden_size])
        self.bi = self.bias_init([self.hidden_size])
        
        self.Wo = self.weight_init([self.hidden_size, self.output_size])
        self.bo = self.bias_init([self.output_size])
        
        self.X = tf.placeholder(tf.float32, shape=(None, None, self.input_size))
        self.Y = tf.placeholder(tf.float32, shape=(None, None, o_size))
        
        self.initial_hidden = tf.matmul(self.X[0, :, :], tf.zeros((self.input_size, self.hidden_size)))
        
    def weight_init(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        
    def bias_init(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def update_state(self, h, X):
        return tf.sigmoid(tf.matmul(X, self.Wx)
                       + tf.matmul(h, self.Wh)
                       + self.bi)
    
    def feed_forward(self, h):
        return tf.sigmoid(tf.matmul(h, self.Wo) + self.bo)
    
    def full_pass_states(self, X):
        states = tf.scan(self.update_state,
                self.X,
                initializer=self.initial_hidden)
        
        return states
    
    def full_pass(self):
        states = self.scan_states()
        outputs = tf.map_fn(self.output, states)
        return outputs
    
    def train(self, features, labels, batch_size, epochs, learning_rate):
        self.inputs = input;
        
        outputs = self.full_pass()
        cross_entropy = -tf.reduce_sum(self.Y * tf.log(outputs) + (1 - self.Y) * tf.log(1 - outputs))
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
        batch = tf.train.batch(self.inputs, batch_size)
        
        for _ in range(epochs):
            for _ in range(self.inputs.shape[0] / batch_size):
                self._X = batch
                train_step
        
