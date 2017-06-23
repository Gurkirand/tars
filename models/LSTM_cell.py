import numpy as np;
import tensorflow as tf;

class LSTM_cell:
    def __init__(self, input_size, hidden_size, output_size, save_path="logs/lstm/", model_name="lstm"):
        self.trained = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.save_path = save_path
        self.model_name = model_name
        
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        self.X = tf.placeholder(tf.float32, shape=(None, 1, input_size))
        self.Y = tf.placeholder(tf.float32, shape=(None, output_size))
        
        self.Wx = tf.get_variable("Wx", [4, input_size, hidden_size], initializer=self.initializer)
        self.Wh = tf.get_variable("Wh", [4, hidden_size, hidden_size], initializer=self.initializer)
        self.b = tf.get_variable("b", [4, hidden_size], initializer=self.initializer)
        
        self.Wo = tf.get_variable("Wo", [hidden_size, output_size], initializer=self.initializer)
        self.bo = tf.get_variable("bo", [output_size], initializer=self.initializer)
        
        _init_h = tf.matmul(self.X[0, :, :], tf.zeros((input_size, hidden_size)))
        _init_c = tf.matmul(self.X[0, :, :], tf.zeros((input_size, hidden_size)))
        self.init_state = tf.stack([_init_h, _init_c])
        self.sess = tf.InteractiveSession()
        
        self.learning_rate = 0.01
    
    def step(self, prev, X):
        h, c = tf.unstack(prev)
        fg = tf.sigmoid(tf.matmul(X, self.Wx[0]) + tf.matmul(h, self.Wh[0]) + self.b[0])
        ig = tf.sigmoid(tf.matmul(X, self.Wx[1]) + tf.matmul(h, self.Wh[1]) + self.b[1])
        cg = tf.sigmoid(tf.matmul(X, self.Wx[2]) + tf.matmul(h, self.Wh[2]) + self.b[2])
        og = tf.sigmoid(tf.matmul(X, self.Wx[3]) + tf.matmul(h, self.Wh[3]) + self.b[3])
        
        c_i = tf.multiply(c, fg) + tf.multiply(ig, cg)
        h_i = tf.multiply(og, tf.tanh(c_i))
        return tf.stack([h_i, c_i])
        
    def train_example(self, example):
        states = tf.scan(self.step,
                         example,
                         self.init_state)
        states = tf.transpose(states, [1, 2, 0, 3])[-1]
        states_reshaped = tf.reshape(states, [-1, self.hidden_size])
        logits = tf.matmul(states_reshaped, self.Wo) + self.bo
        predictions = tf.nn.softmax(logits)
        
        return logits, predictions

    def attempt_restore(self):
        saved = tf.train.get_checkpoint_state(self.save_path)
        saver = tf.train.Saver()
        if (saved and saved.model_checkpoint_path):
            saver.restore(self.sess, saved.model_checkpoint_path)
            self.trained = True
        
    def train(self, features, labels, epochs, learning_rate):
        self.trained = True
        self.learning_rate = learning_rate
        logits, predictions = self.train_example(self.X)
        
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y)
        optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
        self.sess.run(tf.global_variables_initializer())
        self.attempt_restore()
        
        for i in range(0, epochs):
            data = np.transpose(np.array([features, labels]), (1, 0))
            np.random.shuffle(data)
            examples = []
            for j in range(0, len(data)):
                examples.append(data[j])
                
            for example in examples:
                _x, _y = (example)
                _x = np.transpose(np.array([_x]), (1, 0, 2))
                
                p, l, o = self.sess.run([predictions, loss, optimize], feed_dict={self.X:_x, self.Y:_y})
            if (i % 10 == 0):
                print np.average(l)

        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path + self.model_name, global_step=i)
        
    def _generate(self, inits, init_states, length):
        last_out = inits[-1]
        prev = init_states[-1]
        states = []
        outputs = []
        for i in range(length):
            _prev = self.step(prev, last_out)
            states.append(_prev)
            h, c = tf.unstack(_prev)
            outputs.append(tf.nn.softmax(tf.matmul(c, self.Wo) + self.bo))
            last_out = outputs[-1]
            prev = _prev
        return states, outputs
    
    
    def generate(self, length, initial):
        if (not self.trained):
            self.attempt_restore()
            if (not self.trained):
                return
        inits = self.X
        init_states = tf.scan(self.step,
                         inits,
                         self.init_state)
        
        self.sess.run(tf.global_variables_initializer())
        init = np.transpose(initial, (1, 0, 2))
        _i, _s = self.sess.run([inits, init_states], feed_dict={self.X: init})
        
        state = tf.placeholder(tf.float32, (2, 1, self.hidden_size))
        input = tf.placeholder(tf.float32, (1, self.input_size))

        states = []
        outputs = []
        last_out = _i[-1]
        last_state = _s[-1]
        
        _state = self.step(state, input)
        h, c = tf.unstack(_state)
        output = tf.nn.softmax(tf.matmul(c, self.Wo) + self.bo)
        
        for i in range(length):
            ls, lo = self.sess.run([state, output], feed_dict={state: last_state, input: last_out})
            print last_state
            last_state = ls
            last_out = lo
            states.append(ls)
            outputs.append(lo)
        
        os = []
        for o in outputs:
            os.append(np.argmax(o, 1))
        return os
