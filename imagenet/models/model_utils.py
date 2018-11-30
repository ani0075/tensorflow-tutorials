"""
Helper function to create model graph.
"""

import tensorflow as tf


'''
Wrappers for Tensorflow standard operations.
'''
def weight(shape, name):
    initializer = tf.truncated_normal(shape, stddev=0.01)
    w = tf.Variable(initializer, name=name)
    tf.add_to_collection('weights', w)                      # Why?
    return w

def bias(value, shape, name):
    initializer = tf.constant(value, shape=shape)
    return tf.Variable(initializer, name=name)

def conv2d(x, kernel, stride, padding):
    return tf.nn.conv2d(x, kernel, strides=[1, stride[0], stride[1], 1],
        padding=padding)

def max_pool2d(x, kernel_size, stride, padding):
    return tf.nn.max_pool(x, ksize=kernel_size, strides=stride, padding=padding)

def relu(x):
    return tf.nn.relu(x)