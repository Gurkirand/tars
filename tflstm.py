import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
    
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, shape=[None, input_dim])
y = tf.placeholder(tf.float32, shape=[None, output_dim])
last_hidden = tf.placeholder(tf.float32, shape=[None, hidden_dim])

W_1 = weight_variable([input_dim, hidden_dim])
b_1 = bias_variable([hidden_dim])

W_h = weight_variable([hidden_dim, hidden_dim])

l_1 = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1 + tf.matmul(last_hidden, W_h))

W_2 = weight_variable([hidden_dim, output_dim])
b_2 = bias_variable([output_dim])

l_2 = tf.nn.sigmoid(tf.matmul(l_1, W_2) + b_2)

# cost = tf.reduce_sum(tf.square(l_2 - y) / 4)
cross_entropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=l_2))

train_step = tf.train.AdamOptimizer(.1).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())

for _ in range(2000):
    hidden_update = np.zeros((1, hidden_dim))
    f1_int = np.random.randint(largest_number/2) # int version
    f1 = int2binary[f1_int] # binary encoding

    f2_int = np.random.randint(largest_number/2) # int version
    f2 = int2binary[f2_int] # binary encoding

    l_int = f1_int + f2_int
    l = int2binary[l_int]
    
    for pos in range(binary_dim):
        sess.run(train_step, feed_dict={X: [[f1[pos], f2[pos]]], y: [[l[pos]]], last_hidden: hidden_update})
        hidden_update = l_1.eval(feed_dict={X: [[f1[pos], f2[pos]]], y: [[l[pos]]], last_hidden: hidden_update})
        
f1_int = np.random.randint(largest_number/2) # int version
f1 = int2binary[f1_int] # binary encoding

f2_int = np.random.randint(largest_number/2) # int version
f2 = int2binary[f2_int] # binary encoding

# true answer
l_int = f1_int + f2_int
l = int2binary[l_int]
print l_int
hidden_update = np.zeros((1, hidden_dim))
for pos in range(binary_dim):
    print l[pos]
    print l_2.eval(feed_dict={X: [[f1[pos], f2[pos]]], y: [[l[pos]]], last_hidden: hidden_update})
    # print cross_entropy.eval(feed_dict={X: [[f1[pos], f2[pos]]], y: [[l[pos]]], last_hidden: hidden_update})
    hidden_update = l_1.eval(feed_dict={X: [[f1[pos], f2[pos]]], y: [[l[pos]]], last_hidden: hidden_update})
