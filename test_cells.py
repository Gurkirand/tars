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
    
    rnn = RNN_cell(2, 16, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    testrun = rnn.scan_outputs()
    print sess.run(testrun, feed_dict={rnn._X: features})
    


test()
# sess = tf.InteractiveSession()

