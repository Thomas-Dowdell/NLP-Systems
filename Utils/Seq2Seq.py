import functools
import numpy as np
import tensorflow as tf

import util_code as utils
import loss
import optimize

class AttentionCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, cell,
               encoder_output):
    '''
    an RNNCell wrapper for attention mechanism
    implements Bahdanau-style attention, as seen in arXiv:1409.0473
    '''
    self.cell = cell
    self.encoder_output = encoder_output
    
  @property
  def state_size(self):
    return self.cell.state_size

  @property
  def output_size(self):
    return self.cell.output_size
  
  def zero_state(self, batch_size,
                 dtype):
    return self.cell.zero_state(batch_size,
                                dtype)
  
  def __call__(self, inputs,
               state,
               scope = None,
               *args,
               **kwargs):
    with tf.variable_scope('attention'):
      hidden_with_time_axis = tf.expand_dims(state,
                                             axis = 1)
      score = utils.dense(tf.nn.tanh(utils.dense(self.encoder_output,
                                                 output_dim = self.state_size,
                                                 name = 'W1') + utils.dense(hidden_with_time_axis,
                                                                            output_dim = self.state_size,
                                                                            name = 'W2')),
                          output_dim = 1,
                          name = 'V')
      attention_weights = tf.nn.softmax(score,
                                        axis = 1)
      context_vector = attention_weights * self.encoder_output
      context_vector = tf.reduce_sum(context_vector,
                                     axis = 1)
      inputs = tf.concat([inputs, context_vector],
                         axis = 1)
    return self.cell.__call__(inputs,
                              state,
                              scope = scope,
                              *args,
                              **kwargs)

class Seq2Seq():
  def __init__(self, arg,
               name = None):
    '''
    a Seq2Seq model based on the model described in arXiv:1804.00946
    the stop-feature mechanism, in particular, was taken from these mechanisms
    '''
    if name:
      self.name = name
    else:
      self.name = 'Seq2Seq'
    batch_size = 32
    input_sequence_size = 10
    output_sequence_size = 12
    if __name__ != '__main__':
      batch_size = input_sequence_size = output_sequence_size = None
    self.arg = arg
    self.inputs = tf.placeholder(tf.int32,
                                 shape = [batch_size, input_sequence_size],
                                 name = 'inputs')
    self.targets = tf.placeholder(tf.int32,
                                  shape = [batch_size, output_sequence_size],
                                  name = 'targets')
    self.training = tf.placeholder(tf.bool,
                                   name = 'training')
    self.learning_rate = tf.placeholder(tf.float32,
                                        name = 'learning_rate')
    self.keep_prob = tf.placeholder(tf.float32,
                                    name = 'keep_prob')
    self.input_stop_feature = tf.placeholder(tf.float32,
                                             shape = [batch_size, input_sequence_size, 1],
                                             name = 'input_stop_feature')
    self.target_stop_feature = tf.placeholder(tf.float32,
                                              shape = [batch_size, output_sequence_size, 1],
                                              name = 'target_stop_feature')
    self.batch_size = tf.shape(self.inputs)[0]
    self.input_sequence_size = tf.shape(self.inputs)[1]
    self.target_sequence_size = tf.shape(self.targets)[1]
    
    if self.arg.mask_loss:
      self.loss_mask = tf.placeholder(tf.float32,
                                      shape = [batch_size, output_sequence_size], # (batch_size, output_sequence_size)
                                      name = 'loss_mask')
    else:
      self.loss_mask = None
    
    with tf.variable_scope('embedding'):
      embedded_inputs, embedded_targets = self.embedding()
      embedded_inputs = tf.concat([embedded_inputs, self.input_stop_feature],
                                  axis = 2)
      embedded_targets = tf.concat([embedded_targets, self.target_stop_feature],
                                   axis = 2)
    with tf.variable_scope('encode'):
      encoder_output, encoder_state = self.encode(embedded_inputs)
      encoder_output = self.dropout_fn(encoder_output)
    with tf.variable_scope('decode'):
      decoder_output, _ = self.decode(encoder_output,
                                      encoder_state,
                                      embedded_targets)
      decoder_output = self.dropout_fn(decoder_output)
    with tf.variable_scope('output'):
      self.logits = utils.dense(decoder_output,
                                output_dim = self.arg.target_vocab_size,
                                name = 'logits')
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
      self.optimizer.sequential_accuracy(self.logits,
                                         self.targets,
                                         mask = self.loss_mask)
      self.sequential_accuracy = self.optimizer.sequential_accuracy
      self.fetches = [embedded_inputs, encoder_output, self.logits]
      
  def embedding(self):
    self.embed_inputs = tf.get_variable('embed_inputs',
                                        shape = [self.arg.input_vocab_size, self.arg.hidden_dim],
                                        dtype = tf.float32)
    self.embed_targets = tf.get_variable('embed_targets',
                                         shape = [self.arg.target_vocab_size, self.arg.hidden_dim],
                                         dtype = tf.float32)
    return tf.nn.embedding_lookup(self.embed_inputs,
                                  self.inputs), tf.nn.embedding_lookup(self.embed_targets,
                                                                       self.targets)
  
  def encode(self, inputs):
    if self.arg.cell == 'lstm':
      cell = functools.partial(tf.nn.rnn_cell.LSTMCell,
                               num_units = self.arg.hidden_dim)
    elif self.arg.cell == 'gru':
      cell = functools.partial(tf.nn.rnn_cell.GRUCell,
                               num_units = self.arg.hidden_dim)
    cells = []
    for layer in range(1, self.arg.layers + 1):
      cells.append(cell(name = 'cell_{}'.format(layer)))
    self.encode_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    rnn_output, rnn_state = tf.nn.dynamic_rnn(self.encode_cell,
                                              inputs,
                                              dtype = tf.float32)
    return rnn_output, rnn_state
  
  def decode(self, encoder_output,
             encoder_state,
             inputs):
    inputs = tf.concat([tf.zeros([self.batch_size, 1, self.arg.hidden_dim + 1],
                                 dtype = tf.float32), inputs],
                       axis = 1)[:,:-1,:]
    if self.arg.cell == 'lstm':
      cell = functools.partial(tf.nn.rnn_cell.LSTMCell,
                               num_units = self.arg.hidden_dim)
    elif self.arg.cell == 'gru':
      cell = functools.partial(tf.nn.rnn_cell.GRUCell,
                               num_units = self.arg.hidden_dim)
    cells = []
    for layer in range(1, self.arg.layers + 1):
      if self.arg.use_attention:
        cells.append(AttentionCell(cell(name = 'cell_{}'.format(layer)),
                                   encoder_output = encoder_output))
      else:
        cells.append(cell(name = 'cell_{}'.format(layer)))
    self.decode_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    rnn_output, rnn_state = tf.nn.dynamic_rnn(self.decode_cell,
                                              inputs,
                                              dtype = tf.float32,
                                              initial_state = encoder_state)
    return rnn_output, rnn_state
  
  def dropout_fn(self, x,
                 keep_prob = None):
    return tf.cond(self.training,
                   lambda: utils.dropout(x,
                                         keep_prob = self.keep_prob,
                                         dropout = self.arg.dropout_type),
                   lambda: tf.identity(x))
                                              
  
def argument():
  arg = optimize.argument()
  arg.cell = 'gru' # the type of RNN cell. Either GRU or LSTM
  arg.dropout_type = 'vanilla' # dropout type is set to vanilla. There is not SELU activation function, so there is no reason to use alpha-dropout
  arg.loss = 'sparse_softmax_cross_entropy_with_logits' # the loss function used
  arg.stop_feature = 'linear' # the stop_feature used. 'linear' 'tanh' 'exp' 'none'
  
  arg.gamma = 0.1 # the gamma used for the stop-feature
  arg.hidden_dim = 128 # the hidden size
  arg.input_vocab_size = 83 # the input vocab size
  arg.label_smoothing = 1.0 # the label smoothing hyperparameter
  arg.layers = 2 # the number of RNN layers
  arg.target_vocab_size = 120 # the target vocab size
  arg.weight_decay_hyperparameter = 0.001 # the weight decay hyperparameter
  
  arg.mask_loss = True # whether parts of the loss is masked
  arg.use_attention = True # whether the output RNN cells use an attention mechanism 
  arg.weight_decay_regularization = False # whether weight decay is used
  return arg

def stop_feature(batch_size,
                 sequence_size,
                 arg):
  stop_feature = np.ones([batch_size, sequence_size, 1])
  if arg.stop_feature == 'linear':
    for i in range(sequence_size):
      stop_feature[:,i] *= i/sequence_size
  elif arg.stop_feature == 'tanh':
    for i in range(sequence_size):
      stop_feature[:,i] *= np.tanh(arg.gamma * i/sequence_size) + 1 - np.tanh(arg.gamma)
  elif arg.stop_feature == 'exp':
    for i in range(sequence_size):
      stop_feature[:,i] *= np.exp(arg.gamma * (i - sequence_size)/sequence_size)
  elif arg.stop_feature == 'none':
    stop_feature = np.zeros([batch_size, sequence_size, 1])
  return stop_feature
  
if __name__ == '__main__':
  arg = argument()
  
  model = Seq2Seq(arg)