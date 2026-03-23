from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# xavier_initializer replacement (tf.contrib removed in TF2)
xavier_initializer = tf.glorot_uniform_initializer
from contextlib import contextmanager
import numpy as np


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(
        shape=input_layer.get_shape().as_list(),
        mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def scalar_summary(name, x):
    return tf.summary.scalar(name, x)


def histogram_summary(name, x):
    return tf.summary.histogram(name, x)


def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)


def prelu(x, name='prelu', ref=False):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        alpha = tf.get_variable(
            'alpha', in_shape[-1],
            initializer=tf.constant_initializer(0.),
            dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * .5
        if ref:
            return pos + neg, alpha
        else:
            return pos + neg


def conv1d(x, kwidth=5, num_kernels=1, init=None, uniform=False,
           bias_init=None, name='conv1d', padding='SAME'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    assert len(input_shape) >= 3
    w_init = init if init is not None else xavier_initializer()
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kwidth, in_channels, num_kernels],
                            initializer=w_init)
        conv = tf.nn.conv1d(x, W, stride=1, padding=padding)
        if bias_init is not None:
            b = tf.get_variable('b', [num_kernels],
                                initializer=tf.constant_initializer(bias_init))
            conv = conv + b
        return conv


def downconv(x, output_dim, kwidth=5, pool=2, init=None, uniform=False,
             bias_init=None, name='downconv'):
    x2d = tf.expand_dims(x, 2)
    w_init = init if init is not None else xavier_initializer()
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kwidth, 1, x.get_shape()[-1], output_dim],
                            initializer=w_init)
        conv = tf.nn.conv2d(x2d, W, strides=[1, pool, 1, 1], padding='SAME')
        if bias_init is not None:
            b = tf.get_variable('b', [output_dim], initializer=bias_init)
            conv = tf.reshape(tf.nn.bias_add(conv, b), tf.shape(conv))
        else:
            conv = tf.reshape(conv, tf.shape(conv))
        conv = tf.reshape(conv,
            [-1] + [conv.get_shape().as_list()[1]] + [conv.get_shape().as_list()[-1]])
        return conv


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        average_grads.append((grad, v))
    return average_grads


@contextmanager
def variables_on_gpu0():
    old_fn = tf.get_variable
    def new_fn(*args, **kwargs):
        with tf.device("/gpu:0"):
            return old_fn(*args, **kwargs)
    tf.get_variable = new_fn
    yield
    tf.get_variable = old_fn