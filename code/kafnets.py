# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import tensorflow as tf

# Adapted from here:
# https://github.com/ispamm/kernel-activation-functions

def gauss_kernel(x, D, gamma=1.):
    x = tf.expand_dims(x, axis=-1)
    D = tf.reshape(D, (1, 1, -1))
    gauss_kernel = tf.exp(- gamma * tf.square(x - D))
    return gauss_kernel


def kaf(linear, name, kernel='rbf', D=None, gamma=0.1):

    if D is None:
        D = tf.linspace(start=-2., stop=2., num=20)

    with tf.variable_scope('kaf'):
        K = gauss_kernel(linear, D, gamma=gamma)
        alpha = tf.Variable(
            tf.random_normal(
                shape=(1, int(linear.get_shape()[-1]), int(D.get_shape()[0])),
                stddev=0.3,
                ),
            name='alpha'
            )
        act = tf.reduce_sum(tf.multiply(K, alpha), axis=-1)
    return act