import functools
import numpy as np
import tensorflow as tf

import util_code as utils
import loss
import optimize

class RNN():
  def __init__(self, arg,
               name = None):
    '''
    an RNN-model
    '''
    batch_size = 32
    sequence_size = 10
    if __name__ != '__main__':
      batch_size = sequence_size = None
    if name:
      self.name = name
    else:
      self.name = 'RNN'
    self.arg = arg
    self.inputs = tf.placeholder(tf.int32,
                                 shape = [batch_size, sequence_size],
                                 name = 'inputs')
    if self.arg.classification:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size],
                                    name = 'targets')
      self.loss_mask = tf.placeholder(tf.float32,
                                      shape = [batch_size],
                                      name = 'loss_mask')
    else:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size, sequence_size],
                                    name = 'targets')
      self.loss_mask = tf.placeholder(tf.float32,
                                      shape = [batch_size, sequence_size], # (batch_size, sequence_size)
                                      name = 'loss_mask')
    self.keep_prob = tf.placeholder(tf.float32)
    self.training = tf.placeholder(tf.bool)
    self.learning_rate = tf.placeholder(tf.float32)
    self.batch_size = tf.shape(self.inputs)[0]
    self.sequence_size = tf.shape(self.inputs)[1]
    with tf.variable_scope('embedding'):
      embedded_input = self.embedding()
    with tf.variable_scope('RNN'):
      if self.arg.cell == 'lstm':
        cell = functools.partial(tf.nn.rnn_cell.LSTMCell,
                                 num_units = self.arg.hidden_dim)
                                 #dtype = tf.float32)
      elif self.arg.cell == 'gru':
        cell = functools.partial(tf.nn.rnn_cell.GRUCell,
                                 num_units = self.arg.hidden_dim)
                                 #dtype = tf.float32)
      cells = []
      for layer in range(1, self.arg.layers + 1):
        with tf.variable_scope('layer_{}'.format(self.arg.layers)):
          cells.append(cell(name = 'cell_{}'.format(layer)))
          if layer == 1:
            rnn_output, rnn_state = tf.nn.dynamic_rnn(cells[-1],
                                                      embedded_input,
                                                      dtype = tf.float32)
          else:
            if self.arg.unidirectional:
              rnn_output, rnn_state = tf.nn.dynamic_rnn(cells[-1],
                                                        rnn_output,
                                                        dtype = tf.float32)
            else:
              rnn_output, rnn_state = tf.nn.dynamic_rnn(cells[-1],
                                                        rnn_output,
                                                        initial_state = rnn_state)
    with tf.variable_scope('output'):
      if self.arg.classification:
        if self.arg.cell == 'lstm':
          rnn_state = rnn_state[-1]
        elif self.arg.cell == 'gru':
          rnn_state = rnn_state
        self.logits = self.output(rnn_state)
      else:
        self.logits = self.output(rnn_output)
        
    with tf.variable_scope('loss'):
      self.loss_cl = loss.Loss(self.logits,
                               self.targets,
                               self.arg.loss,
                               vocab_size = self.arg.target_vocab_size,
                               label_smoothing = self.arg.label_smoothing)
      cost = self.loss_cl.loss
      if self.arg.mask_loss:
        self.cost = tf.reduce_mean(cost * self.loss_mask)
      else:
        self.cost = tf.reduce_mean(cost)
      if self.arg.weight_decay_regularization:
        l2_loss = self.loss_cl.l2_loss(tf.trainable_variables())
        l2_loss *= self.arg.weight_decay_hyperparameter
        self.cost += l2_loss
      self.optimizer = optimize.Optimizer(arg,
                                          loss = self.cost,
                                          learning_rate = self.learning_rate)
      self.optimizer.accuracy(self.logits,
                              self.targets,
                              mask = self.loss_mask)
      self.train_op = self.optimizer.train_op
      self.predict = self.optimizer.predict
      self.correct_prediction = self.optimizer.correct_prediction
      self.accuracy = self.optimizer.accuracy
    
  def embedding(self):
    self.embed_weights = tf.get_variable('embed',
                                         shape = [self.arg.input_vocab_size, self.arg.hidden_dim],
                                         dtype = tf.float32)
    return tf.nn.embedding_lookup(self.embed_weights,
                                  self.inputs)
  
  def output(self, rnn_output):
    with tf.variable_scope('output'):
      weights = tf.get_variable('weights',
                                shape = [self.arg.hidden_dim, self.arg.target_vocab_size],
                                dtype = tf.float32)
      bias = tf.get_variable('bias',
                             shape = [self.arg.target_vocab_size],
                             dtype = tf.float32)
    return tf.tensordot(rnn_output,
                        weights,
                        axes = 1) + bias
  
def argument():
  arg = optimize.argument()
  arg.cell = 'gru' # whether to use a GRU or LSTM model
  arg.loss = 'sparse_softmax_cross_entropy_with_logits'
  
  arg.hidden_dim = 256 # the hidden size of the model
  arg.label_smoothing = 1.0 # the hyperparameter for smoothing labels
  arg.layers = 2 # the number of layers for RNN
  arg.input_vocab_size = 1000 # the vocab size of the input sequence
  arg.target_vocab_size = 1000 # the target size of the target sequence
  arg.weight_decay_hyperparameter = 0.001 # the hyperparameter for weight decay
  
  arg.classification = True # whether the output is a sequence or a single token
  arg.mask_loss = True # whether to mask parts of the loss
  arg.unidirectional = True # whether the anaylsis is strictly unidirectional
  arg.weight_decay_regularization = False # whether weight decay is used
  
  arg.hidden_size = arg.hidden_dim
  return arg
  
if __name__ == '__main__':
  arg = argument()
  
  model = RNN(arg)