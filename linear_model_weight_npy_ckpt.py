'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import os
import numpy as np
rng = numpy.random
weight_dir = './weight_part'
# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)
# get variable and save list
all_var = tf.trainable_variables()
varlist = []
for var in all_var:
    varlist.append(var)
var_dict = {}
init_ops = []
outfile = "./weight.npy"
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(weight_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(weight_dir, ckpt_name))
        print('[*] Success to read {}'.format(ckpt_name))
        # exit(0) #verify load only part work
    else:
        print('[*] Failed to find a checkpoint. Start training from scratch ...')

    #compare load ckpt and load npy
    w_ck = sess.run(W)
    b_ck = sess.run(b)
    ####load part, yes, it can be done, once the name is same
    if True:
        var_dict = np.load(outfile).item()
        for var in tf.trainable_variables():
            value = var_dict[var.name]
            init_op = tf.assign(var, value, validate_shape=True)
            init_ops.append(init_op)
        # debugging:
        for init_op in init_ops:
            sess.run(init_op)
        w_npy = sess.run(W)
        b_npy = sess.run(b)
        print("compare",w_ck,w_npy,b_ck,b_npy)
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))
    #save weight
    if False:#freeze weight
        print("save weight")
        weight_name = '/linear_model' + '.ckpt'
        saver.save(sess, weight_dir + weight_name)
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    #get all variable's values and save them in a dictionary
    for var in varlist:
        value = sess.run(var)
        var_dict[var.name] = value
    #

    np.save(outfile, var_dict)
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()