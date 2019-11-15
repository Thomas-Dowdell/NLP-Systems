import numpy as np
import tensorflow as tf

import util_code as utils

def next_state(cur_state,
               args_tup):
  cur_x_times_one_minus_f, cur_f = args_tup
  return cur_f * cur_state + cur_x_times_one_minus_f

def SRU(x,
        num_layers = 2,
        activation = None,
        initial_state = None,
        name = None,
        reuse = None,
        reuse_layer = False):
  '''
  SRU introduced in arXiv:1709.02755
  code based on tensor2tensor
  x - tensor, dtype = tf.float32 shape = [batch_size, sequence_size, hidden_size]
  '''
  with tf.variable_scope(name,
                         default_name = 'SRU',
                         reuse = reuse):
    tf_x_shape = tf.shape(x)
    x_shape = x.shape.as_list()
    x = tf.transpose(x,
                     perm = [1, 0, 2],
                     name = 'input_transpose')
    if initial_state is None:
      initial_state = tf.zeros([tf.shape(x)[1], tf.shape(x)[2]],
                               dtype = x.dtype)
    for i in range(num_layers):
      with tf.variable_scope('layer_{}'.format(i + 1),
                             reuse = (i != 0 and reuse_layer)):
        x_orig = x
        x, f, r = tf.split(utils.dense(x,
                                       output_dim = 3 * x_shape[-1],
                                       name = 'dense'),
                           num_or_size_splits = 3,
                           axis = -1)
        f, r = tf.sigmoid(f), tf.sigmoid(r)
        x_times_one_minus_f = x * (1.0 - f)
        c_states = tf.scan(next_state,
                           (x_times_one_minus_f, f),
                           initializer = initial_state,
                           parallel_iterations = 2,
                           name = 'scan_{}'.format(i))
        if activation is not None:
          c_states = activation(c_states)
        h = c_states * r + (1.0 - r) * x_orig
        x = h
    x = tf.transpose(x,
                     perm = [1, 0, 2])
    return tf.reshape(x, tf_x_shape)