import numpy as np
import tensorflow as tf

import util_code as utils
import loss
import optimize
import develop_bias

class ACT():
  def __init__(self, batch_size,
               sequence_size):
    '''
    an adaptive-computational-time mechanism, as introduced in arXiv:1603.08983
    used in Universal Transformer, arXiv:1807.03819
    based off code from tensor2tensor
    '''
    self.halting_probability = tf.zeros([batch_size, sequence_size],
                                        name = 'halting_probability')
    self.remainders = tf.zeros([batch_size, sequence_size],
                               name = 'remainder')
    self.n_updates = tf.zeros([batch_size, sequence_size],
                              name = 'n_updates')
    
  def __call__(self, pondering,
               halt_threshold,
               halting_probability,
               remainders,
               n_updates):
    still_running = tf.cast(tf.less(halting_probability,
                                    1.0),
                            dtype = tf.float32)
    new_halted = tf.greater(halting_probability + pondering * still_running,
                            halt_threshold)
    new_halted = tf.cast(new_halted,
                         dtype = tf.float32) * still_running
    still_running_now = tf.less_equal(halting_probability + pondering * still_running,
                                      halt_threshold)
    still_running_now = tf.cast(still_running_now,
                                dtype = tf.float32) * still_running
    halting_probability += pondering * still_running
    remainders += new_halted * (1 - halting_probability)
    halting_probability += new_halted * remainders
    n_updates += still_running + new_halted
    update_weights = pondering * still_running + new_halted * remainders
    update_weights = tf.expand_dims(update_weights,
                                    axis = -1)
    return update_weights, halting_probability, remainders, n_updates
  
  def should_continue(self, threshold):
    return tf.reduce_any(tf.less(self.halting_probability,
                                 threshold))