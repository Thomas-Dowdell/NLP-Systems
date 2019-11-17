import numpy as np
import tensorflow as tf

import sys
sys.path.append('/home/tom/Desktop/Programming/Models/Utils')
import util_code as utils
import loss
import optimize

class Neural_GPU():
  def __init__(self, arg,
               name = None):
    if name:
      self.name = name
    else:
      self.name = 'Neural-GPU'
    self.arg = arg
    
    batch_size = 128
    sequence_size = 10
    if __name__ != '__main__':
      batch_size = sequence_size = None
    self.inputs = tf.placeholder(tf.int32,
                                 shape = [batch_size, sequence_size])
    self.targets = tf.placeholder(tf.int32,
                                  shape = [batch_size, sequence_size])
    self.batch_size = tf.shape(self.inputs)[0]
    self.sequence_size = tf.shape(self.inputs)[1]
    self.training = tf.placeholder(tf.bool)
    self.keep_prob = tf.placeholder(tf.float32)
    self.learning_rate = tf.placeholder(tf.float32)
    self.embed_weight = tf.get_variable('embed_weight',
                                        shape = [self.arg.vocab_size, self.arg.embed_dim])
    s0 = tf.nn.embedding_lookup(self.embed_weight,
                                self.inputs)
    s0 = tf.reshape(s0,
                    shape = [self.batch_size, 1, sequence_size, self.arg.embed_dim])
    s0 = tf.concat([s0, tf.zeros([self.batch_size, self.arg.width - 1, sequence_size, self.arg.embed_dim])],
                   axis = 1)
    sfin = self.neural_gpu(s0)
    output = sfin[:,0,:,:]
    self.output_weight = tf.get_variable('output_weight',
                                         shape = [self.arg.embed_dim, self.arg.vocab_size])
    self.output_bias = tf.get_variable('output_bias',
                                       shape = [self.arg.vocab_size])
    self.output = tf.tensordot(output,
                               self.output_weight,
                               axes = 1) + self.output_bias
    self.predict = tf.argmax(self.output,
                             axis = -1,
                             output_type = tf.int32)
    self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.one_hot(self.targets,
                                                                               depth = self.arg.vocab_size),
                                                           logits = self.output,
                                                           axis = -1)
    self.loss = tf.reduce_mean(self.loss)
    self.optimizer = optimize.Optimizer(arg,
                                        loss = self.loss,
                                        learning_rate = self.learning_rate)
    self.optimizer.accuracy(self.output,
                            self.targets)
    self.train_op = self.optimizer.train_op
    self.predict = self.optimizer.predict
    self.correct_prediction = self.optimizer.correct_prediction
    self.accuracy = self.optimizer.accuracy
    
  def neural_gpu(self, s):
    def neural_gpu(i, s):
      for l in range(self.arg.layer):
        with tf.variable_scope('layer_{}'.format(l + 1)):
          s = self.cgru(s)
      s = self.dropout_fn(s)
      return i + 1, s
    i = 0
    cond = lambda i, s: tf.less(i, self.sequence_size)
    body = lambda i, s: neural_gpu(i, s)
    i, s = tf.while_loop(cond = cond,
                         body = body,
                         loop_vars = [i, s])
    return s
  
  def dropout_fn(self, x,
                 keep_prob = None):
    return tf.cond(self.training,
                   lambda: utils.dropout(x,
                                         keep_prob = self.keep_prob,
                                         dropout = self.arg.dropout_type),
                   lambda: tf.identity(x))
    
  def update_gate(self, s):
    with tf.variable_scope('update_gate'):
      weights = tf.get_variable('weight',
                                shape = [self.arg.kernel_width, self.arg.kernel_height, self.arg.embed_dim, self.arg.embed_dim])
      bias = tf.get_variable('bias',
                             shape = [self.arg.embed_dim])
      return tf.nn.convolution(input = s,
                               filter = weights,
                               padding = 'SAME') + bias
    
  def reset_gate(self, s):
    with tf.variable_scope('reset_gate'):
      weights = tf.get_variable('weight',
                                shape = [self.arg.kernel_width, self.arg.kernel_height, self.arg.embed_dim, self.arg.embed_dim])
      bias = tf.get_variable('bias',
                             shape = [self.arg.embed_dim])
      return tf.nn.convolution(input = s,
                               filter = weights,
                               padding = 'SAME') + bias
    
  def cgru(self, s):
    weights = tf.get_variable('weight',
                              shape = [self.arg.kernel_width, self.arg.kernel_height, self.arg.embed_dim, self.arg.embed_dim])
    bias = tf.get_variable('bias',
                           shape = [self.arg.embed_dim])
    update = tf.nn.sigmoid(self.update_gate(s))
    reset = tf.nn.sigmoid(self.reset_gate(s))
    return tf.multiply(update, 
                       s) + tf.multiply(1 - update,
                                        tf.nn.tanh(tf.nn.convolution(input = tf.multiply(s,
                                                                                         reset),
                                                                     filter = weights,
                                                                     padding = 'SAME') + bias))
  
  def sigmoid_cutoff(self, s):
    return tf.maximum(0, 
                      tf.minimum(1,
                                 1.2 * tf.nn.sigmoid(s) - 0.1))
  
def argument():
  arg = optimize.argument()
  
  arg.dropout_type = 'vanilla'
  
  arg.embed_dim = 24
  arg.kernel_height = 1
  arg.kernel_width = 3
  arg.layer = 2
  arg.vocab_size = 20
  arg.width = 3
  return arg

if __name__ == '__main__':
  arg = argument()
  
  model = Neural_GPU(arg)