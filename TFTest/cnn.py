from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# This script is meant to train MNIST data through a Convolutional Neural Network (CNN).


# Initializes weights.
# small amount of noises for symmetry breaking and to prevent 0 gradients
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# ReLU neurons => slightly positive initial bias to avoid dead neurons
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)	

# Stride of one with zero padding => output and input have same size
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# pooling: max pooling over 2x2 blocks
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

# load data set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

# Placeholders.
# 28 x 28 MNIST => 784 pixels
# y_ is a 2d tensor s.t. each row is a one-hot vector in R^10.
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Variables - 
# W is a 784 x 10 matrix (784 inputs, 10 outputs)
# b is a 10-dimensional vector (due to 10 classes)
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


'''
First convolutional layer, followed by max pooling
	- Computes 32 features for each 5x5 patch
	- weight tensor: [5, 5, 1, 32]
		- dimensions: patch size, input channels, output channels
	- include bias vector with comp. for each output channel	
'''
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# reshape x -> 4D tensor, with:
# 	image width corresponding to 2nd dimension
# 	height corresponding to 3rd dimension
#	color channels corresponding to 4th dimension
x_image = tf.reshape(x, [-1, 28, 28, 1])

# convolve x_image with weight tensor, add bias, apply ReLU function,
# max pool => reduce image size to 14x14
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# SECOND convolutional layer - 64 features for 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# add fully-connected layer with 1024 neurons => process entire image
# also reshape tensor from pooling layer -> batch of vectors
# then multiply by weight matrix, add bias, apply ReLU
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
DROPOUT: reduce overfitting by adding such before readout layer
create placeholder for probability that neuron output is kept during dropout
	=> allows us to turn dropout on during training, off during testing

tf.nn.dropout automatically handles scaling neuron outputs and masking
'''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# regression model, multiply vectorized input images x by W + bias
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# loss function indicates how bad model's prediction on simple ex
# therefore minimize such loss with cross-entropy between target and softmax
# activation function applied to model's prediction.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=y_conv))

# Training with ADAM optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

'''
evaluation
tf.argmax gives index of highest entry in tensor along given axis
 	ex: tf.argmax(y, 1) => label model thinks is most likely for each input
 		tf.argmax(y_, 1) => TRUE label
 		tf.equal: check if a = b, s.t. a and b are parameters
'''
# returns bool
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(2000):
  batch = mnist.train.next_batch(50)
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
