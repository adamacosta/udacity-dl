"""
From the tensorflow MNIST for ML beginners tutorial:

https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Data is divided into three sets: mnist.train (55,000 images),
# mnist.test (10,000 images), and mnist.validaton (5,000 images).
#
# Each dataset is divided into images and labels i.e. mnist.train.images
# and mnist.train.labels. Each image is a 28 x 28 matrix of values
# between 0 and 255 giving a pixel intensity. Each matrix is flattened
# into a 28 * 28 = 784 entry row, so that ultimately our training set 
# is a 2-dimensional, [55000, 784] tensor, with the first dimension
# giving the image index and the second giving all of the pixels.
#
# Each label is an integer in the range [0, 9] encoded into a one-hot
# indicator matrix.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create a placeholder for each batch with a first dimension of 'None'
# which can be of any batch size.
x = tf.placeholder(tf.float32, [None, 784])

# Create variables to hold the weights and biases of the softmax model.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Our response model is the linear output filtered through a softmax
# activation function to turn evidence weight into a probability distribution.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# The model is trained by simple gradient descent to minimize cross-entropy
# between the predicted and true distributions.
#
# First, we create a placeholder for truth values, again with a first
# dimension of 'None' to allow for any batch size.
y_ = tf.placeholder(tf.float32, [None, 10])

# Then, we create the cross-entropy loss function.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Then, we create the graph node defining a training step.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Now that the model is set up, we create a graph node to initialize all variables.
init = tf.initialize_all_variables()

# Then, launch a session to run the computation graph.
sess = tf.Session()
sess.run(init)

# We'll try 1000 batches of 100 images each
# The computation graph defined above only requires the image batch and 
# the corresponding label batch for each train_step, with everything else
# being a computed value, so we give those to the feed dict to train_step.
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# This completes the training and we can evaluate the model.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
