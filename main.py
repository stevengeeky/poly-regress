"""
A Test Training Session
"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def main():
    """
    Main evaluation
    """
    
    sess = tf.InteractiveSession()
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    tf.global_variables_initializer().run()
    
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    entropy = tf.reduce_mean(-tf.reduce_sum( y_ * tf.log(y), reduction_indices=[1] ))
    
    step = optimizer.minimize(entropy)
    
    for i in range(1000):
        print("%d " % i),
        batch = mnist.train.next_batch(100)
        step.run(feed_dict={ x: batch[0], y_: batch[1] })
    
    print "\n"
    
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    evaluation_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    
    print "Accuracy: %f" % evaluation_accuracy

main()
