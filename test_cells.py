from RNN_cell import RNN_cell
import numpy as np
import tensorflow as tf

int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
        
def init_binary_dataset(size):
    features = []
    labels = []
    for i in range(10):
        # generate a simple addition problem (a + b = c)
        f1_int = np.random.randint(largest_number/2) # int version
        f1 = int2binary[f1_int] # binary encoding

        f2_int = np.random.randint(largest_number/2) # int version
        f2 = int2binary[f2_int] # binary encoding

        # true answer
        l_int = f1_int + f2_int
        l = int2binary[l_int]
        features.append([f1, f2])
        labels.append([l])
    return np.array(features), np.array(labels)

def test():
    features, labels = init_binary_dataset(10)
    
    print "Features shape: ", features.shape
    print "Labels shape: ", labels.shape
    
    features = np.reshape(features, (8, 10, 2))
    labels = np.reshape(labels, (8, 10, 1))
    
    rnn = RNN_cell(2, 16, 1)

    input = tf.placeholder(tf.float32, [None, None, 2])
    output = tf.placeholder(tf.float32, [None, None, 1])
    train = rnn.train(input, output, 10, 1, 0.1)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # print sess.run(rnn.train, feed_dict={features: features, labels: labels, batch_size: 10, epochs: 1, learning_rate:0.1})
    print sess.run(train, feed_dict={input: features, output:labels})
    


test()
# sess = tf.InteractiveSession()

