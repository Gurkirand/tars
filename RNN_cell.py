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
        self.Y = tf.placeholder(tf.float32, shape=(None, None, self.output_size))
        
        self.initial_hidden = tf.matmul(self.X[0, :, :], tf.zeros((self.input_size, self.hidden_size)))
        self.sess = tf.InteractiveSession()
        
    def weight_init(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        
    def bias_init(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def update_state(self, h, X):
        return tf.tanh(tf.matmul(X, self.Wx)
                       + tf.matmul(h, self.Wh)
                       + self.bi)
    
    def feed_forward(self, h):
        return tf.nn.softmax(tf.matmul(h, self.Wo) + self.bo)
    
    def feed_forward_logits(self, h):
        return tf.matmul(h, self.Wo) + self.bo
    
    def full_pass_states(self):
        states = tf.scan(self.update_state,
                self.X,
                initializer=self.initial_hidden)
        
        return states
    
    def full_pass(self):
        states = self.full_pass_states()
        outputs = tf.map_fn(self.feed_forward_logits, states)
        return outputs
    
    def train(self, features, labels, batch_size, epochs, learning_rate):
        outputs = self.full_pass()
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=self.Y)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        

        data_size = features.shape[0]
        batches = []
        
        self.sess.run(tf.global_variables_initializer())
        for i in range(0, data_size - 1, batch_size - 1):
            batches.append([features[i:i + (batch_size - 1)], labels[i:i + (batch_size - 1)]])
        
        for i in range(epochs):
            for input, output in batches:
                 o, l, t = self.sess.run([outputs, loss, train_step], feed_dict={self.X: input, self.Y: output})
            print np.average(l)
        
    def run(self, input, length):
        input = np.array([input])
        outputs = []
        h = self.initial_hidden
        hidden = self.update_state(h, self.X[0])
        output = self.feed_forward(hidden)
        for _ in range(length):
            h, o = self.sess.run([hidden, output], feed_dict={self.X: input})
            outputs.append(np.argmax(o, axis=-1))
            input = np.array([o])
        return outputs
            
