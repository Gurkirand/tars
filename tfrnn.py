import numpy as np
import tensorflow as tf

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
feature1 = tf.placeholder(tf.float32, shape=[None, binary_dim])
feature2 = tf.placeholder(tf.float32, shape=[None, binary_dim])
label = tf.placeholder(tf.float32, shape=[None, binary_dim])

init_state = tf.get_variable("init_state", initializer=tf.zeros([1, hidden_dim]))

initializer = tf.random_normal_initializer(stddev=0.1)
inputs = tf.stack([feature1, feature2], axis=2)

with tf.variable_scope("RNN") as scope:
    h = init_state
    ys = []
    for i, X in enumerate(tf.split(inputs, 8, axis=1)):
        if i > 0: scope.reuse_variables()
        X = tf.reshape(X, [-1, 2])
        Whx = tf.get_variable("Whx", [input_dim, hidden_dim], initializer=initializer)
        Whh = tf.get_variable("Whh", [hidden_dim, hidden_dim], initializer=initializer)
        bh = tf.get_variable("bh", [hidden_dim], initializer=initializer)
        Why = tf.get_variable("Why", [hidden_dim, output_dim], initializer=initializer)
        by = tf.get_variable("byy", [output_dim], initializer=initializer)
        
        h = tf.nn.sigmoid(tf.matmul(X, Whx) + tf.matmul(h, Whh) + bh)
        y = tf.nn.sigmoid(tf.matmul(h, Why) + by)
        ys.insert(0, y)
    
outputs = tf.concat(ys, axis=1)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=outputs))

minimizer = tf.train.AdamOptimizer(0.1)
train_step = minimizer.minimize(loss)

sess.run(tf.global_variables_initializer())

for _ in range(1000):
    f1_int = np.random.randint(largest_number/2) # int version
    f1 = [int2binary[f1_int]] # binary encoding

    f2_int = np.random.randint(largest_number/2) # int version
    f2 = [int2binary[f2_int]] # binary encoding

    l_int = f1_int + f2_int
    l = [int2binary[l_int]]
    
    sess.run([train_step],
             feed_dict = {feature1 : f1,
                          feature2 : f2,
                          label : l})
    
                    
    
f1_int = np.random.randint(largest_number/2) # int version
f1 = [int2binary[f1_int]] # binary encoding

f2_int = np.random.randint(largest_number/2) # int version
f2 = [int2binary[f2_int]] # binary encoding

l_int = f1_int + f2_int
l = [int2binary[l_int]]
print l
print sess.run([outputs],
         feed_dict = {feature1 : f1,
                      feature2 : f2,
                      label : l})
