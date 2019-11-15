import numpy as np
import tensorflow as tf

import util_code as utils
import loss
import optimize
import develop_bias

from ACT import ACT

class Universal_Transformer():
  def __init__(self, arg,
               name = None):
    '''
    the Universal Transformer, introduced in arXiv:1807.03819
    based off code from tensor2tensor
    '''
    if name:
      self.name = name
    else:
      self.name = 'Universal_Transformer'
    self.arg = arg
    batch_size = 32
    input_sequence_size = 10
    output_sequence_size = 12
    if not self.arg.use_decoder:
      output_sequence_size = input_sequence_size
    if __name__ != '__main__':
      batch_size = input_sequence_size = output_sequence_size = None
    self.inputs = tf.placeholder(tf.int32,
                                 shape = [batch_size, input_sequence_size],
                                 name = 'inputs')
    if self.arg.classification:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size], # (batch_size, output_sequence_size)
                                    name = 'targets')
    else:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size, output_sequence_size], # (batch_size, output_sequence_size)
                                    name = 'targets')
    self.training = tf.placeholder(tf.bool)
    self.keep_prob = tf.placeholder(tf.float32)
    self.learning_rate = tf.placeholder(tf.float32)
    self.batch_size = tf.shape(self.inputs)[0]
    self.input_sequence_size = tf.shape(self.inputs)[1]
    if not self.arg.classification:
      self.target_sequence_size = tf.shape(self.targets)[1]
    
    self.encoder_self_attention_bias = develop_bias._create_mask(self.input_sequence_size,
                                                                 self.arg.unidirectional_encoder)
    if not self.arg.classification:
      self.encoder_decoder_attention_bias = tf.zeros([1, 1, self.target_sequence_size, self.input_sequence_size],
                                                     name = 'encoder_self_attention_bias')
      self.decoder_self_attention_bias = develop_bias._create_mask(self.target_sequence_size,
                                                                   self.arg.unidirectional_decoder)
    
    if self.arg.mask_loss:
      if self.arg.classification:
        self.loss_mask = tf.placeholder(tf.float32,
                                        shape = [batch_size],
                                        name = 'loss_mask')
      else:
        self.loss_mask = tf.placeholder(tf.float32,
                                        shape = [batch_size, output_sequence_size], # (batch_size, output_sequence_size)
                                        name = 'loss_mask')
    else:
      self.loss_mask = None
      
    self.ffd = self.transformer_ffd
      
    if 'stop' in self.arg.pos:
      embedding_size = self.arg.hidden_size - 1
    else:
      embedding_size = self.arg.hidden_size
    with tf.variable_scope('encoder_embedding'):
      encoder_input, enc_params = utils.embedding(self.inputs,
                                                  model_dim = embedding_size,
                                                  vocab_size = self.arg.input_vocab_size,
                                                  name = 'encode')
    if not self.arg.classification:
      with tf.variable_scope('decoder_embedding'):
        decoder_input, dec_params = utils.embedding(self.targets,
                                                    model_dim = embedding_size,
                                                    vocab_size = self.arg.target_vocab_size,
                                                    name = 'decode')
      if self.arg.use_decoder:
        params = dec_params
        del enc_params
      else:
        params = enc_params
        del dec_params
    else:
      params = enc_params
    
    with tf.variable_scope('encoder'):
      encoder_input = self.dropout_fn(encoder_input)
      encoder_output, enc_n_updates, enc_remainders = self.encoder(encoder_input,
                                                                   encoder_self_attention_bias = self.encoder_self_attention_bias)
      enc_act_loss = tf.reduce_mean(enc_n_updates + enc_remainders)
    if arg.use_decoder:
      with tf.variable_scope('decoder'):
        decoder_input = tf.pad(decoder_input,
                               paddings = [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        decoder_input = self.dropout_fn(decoder_input)
        decoder_output, dec_n_updates, dec_remainders = self.decoder(decoder_input,
                                                                     encoder_output,
                                                                     decoder_self_attention_bias = self.decoder_self_attention_bias,
                                                                     encoder_decoder_attention_bias = self.encoder_decoder_attention_bias)
        dec_act_loss = tf.reduce_mean(dec_n_updates + dec_remainders)
    else:
      dec_act_loss = 0.0
    if self.arg.classification:
      if self.arg.use_decoder:
        decoder_output = decoder_output[:,-1]
      else:
        encoder_output = encoder_output[:,-1]
    with tf.variable_scope('output'):
      if self.arg.use_mos:
        if self.arg.use_decoder:
          self.logits = mos.MoS(decoder_output,
                                hidden_size = self.arg.hidden_size,
                                vocab_size = self.arg.target_vocab_size)
        else:
          self.logits = mos.MoS(encoder_output,
                                hidden_size = self.arg.hidden_size,
                                vocab_size = self.arg.target_vocab_size)
        self.logits = tf.nn.softmax(self.logits)
        if self.arg.loss == 'sparse_softmax_cross_entropy_with_logits':
          self.arg.loss = 'log_loss'
        self.loss_cl = loss.Loss(self.logits,
                                 self.targets,
                                 self.arg.loss,
                                 vocab_size = self.arg.target_vocab_size,
                                 activation = tf.identity,
                                 label_smoothing = self.arg.label_smoothing)
        cost = tf.reduce_sum(self.loss_cl.loss,
                             axis = -1)
      else:
        #weights = tf.transpose(params[0],
        #                       [1, 0])
        weights = tf.get_variable('weights',
                                  shape = [self.arg.hidden_size, self.arg.target_vocab_size],
                                  dtype = tf.float32)
        bias = tf.get_variable('bias',
                               shape = [self.arg.target_vocab_size],
                               dtype = tf.float32)
        if arg.use_decoder:
          self.logits = tf.tensordot(decoder_output,
                                     weights,
                                     axes = 1) + bias
        else:
          self.logits = tf.tensordot(encoder_output,
                                     weights,
                                     axes = 1) + bias
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
    if self.arg.use_act:
      self.cost += (enc_act_loss + dec_act_loss) * self.arg.act_loss_weight
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
      
  def encoder(self, encoder_input,
              encoder_self_attention_bias):
    act = ACT(self.batch_size,
              self.input_sequence_size)
    halt_threshold = 1.0 - self.arg.act_epsilon
    
    def cond(x, 
             i, 
             halting_probability, 
             remainders, 
             n_updates):
      if self.arg.use_act:
        return tf.reduce_all([act.should_continue(halt_threshold), tf.less(i, self.arg.max_encoder_steps)])
      else:
        return tf.less(i,
                       self.arg.max_encoder_steps)
    
    def body(x, 
             i,
             halting_probability, 
             remainders, 
             n_updates):
      with tf.variable_scope('encoder_layer'):
        state = x
        x += self.timing_position(x)
        pondering = utils.dense(x,
                                output_dim = 1,
                                name = 'pondering')
        pondering = tf.squeeze(pondering,
                               axis = -1)
        pondering = tf.nn.sigmoid(pondering)
        update_weights, halting_probability, remainders, n_updates = act(pondering,
                                                                         halt_threshold, 
                                                                         halting_probability, 
                                                                         remainders, 
                                                                         n_updates)
        with tf.variable_scope('attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = None,
                                        bias = encoder_self_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        deparameterize = self.arg.deparameterize,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = self.arg.relative_attention,
                                        max_relative_position = self.arg.max_relative_position)
          y = self.dropout_fn(y)
          x += y    
        with tf.variable_scope('ffd'):
          y = utils.layer_norm(x)
          y = self.ffd(y)
          y = self.dropout_fn(y)
          x += y
    
        x = (x * update_weights) + (state * (1 - update_weights))
      return x, i + 1, halting_probability, remainders, n_updates
    
    encoder_output, _, _, remainders, n_updates = tf.while_loop(cond,
                                                                body,
                                                                [encoder_input, 0, act.halting_probability, act.remainders, act.n_updates])
    with tf.variable_scope('output'):
      return utils.layer_norm(encoder_output), n_updates, remainders
  
  def decoder(self, inputs,
              memory,
              decoder_self_attention_bias,
              encoder_decoder_attention_bias):
    act = ACT(self.batch_size,
              self.target_sequence_size)
    halt_threshold = 1.0 - self.arg.act_epsilon
    
    def cond(x, 
             i, 
             halting_probability, 
             remainders, 
             n_updates):
      if self.arg.use_act:
        return tf.reduce_all([act.should_continue(halt_threshold), tf.less(i, 
                                                                           self.arg.max_decoder_steps)])
      else:
        return tf.less(i, self.arg.max_decoder_steps)
    
    def body(x, 
             i,
             halting_probability, 
             remainders, 
             n_updates):
      with tf.variable_scope('decoder_layer'):
        state = x
        x += self.timing_position(x)
        pondering = utils.dense(x,
                                output_dim = 1,
                                name = 'pondering')
        pondering = tf.squeeze(pondering,
                               axis = -1)
        pondering = tf.nn.sigmoid(pondering)
        update_weights, halting_probability, remainders, n_updates = act(pondering,
                                                                         halt_threshold, 
                                                                         halting_probability, 
                                                                         remainders, 
                                                                         n_updates)
        with tf.variable_scope('attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = None,
                                        bias = decoder_self_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        deparameterize = self.arg.deparameterize,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = self.arg.relative_attention,
                                        max_relative_position = self.arg.max_relative_position)
          y = self.dropout_fn(y)
          x += y   
        with tf.variable_scope('encoder_attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = memory,
                                        bias = encoder_decoder_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = False,
                                        max_relative_position = self.arg.max_relative_position)
          y = self.dropout_fn(y)
          x += y
        with tf.variable_scope('ffd'):
          y = utils.layer_norm(x)
          y = self.ffd(y)
          y = self.dropout_fn(y)
          x += y
    
        x = (x * update_weights) + (state * (1 - update_weights))
      return x, i + 1, halting_probability, remainders, n_updates
    
    decoder_output, _, _, remainders, n_updates = tf.while_loop(cond,
                                                                body,
                                                                [inputs, 0, act.halting_probability, act.remainders, act.n_updates])
    with tf.variable_scope('output'):
      return utils.layer_norm(decoder_output), n_updates, remainders
  
  def transformer_ffd(self, x):
    x = utils.dense(x,
                    output_dim = self.arg.filter_size,
                    use_bias = True,
                    name = 'ffd_1')
    x = self.dropout_fn(x)
    if self.arg.use_relu:
      x = tf.nn.relu(x)
    else:
      x = utils.gelu(x)
    return utils.dense(x,
                       output_dim = self.arg.hidden_size,
                       use_bias = True,
                       name = 'ffd_2')
      
  def dropout_fn(self, x,
                 keep_prob = None):
    return tf.cond(self.training,
                   lambda: utils.dropout(x,
                                         keep_prob = self.keep_prob,
                                         dropout = self.arg.dropout_type),
                   lambda: tf.identity(x))
          
  def timing_position(self, inputs):
    sequence_size = tf.shape(inputs)[1]
    
    if self.arg.pos == 'timing':
      return inputs + utils.add_timing_signal_1d(sequence_size = sequence_size,
                                                 channels = self.arg.hidden_size)
    elif self.arg.pos == 'emb':
      return inputs + utils.add_positional_embedding(inputs,
                                                     max_length = self.arg.input_max_length, ###
                                                     hidden_size = self.arg.hidden_size,
                                                     input_sequence_size = sequence_size,
                                                     name = 'positional_embedding')
    elif self.arg.pos == 'linear_stop':
      sequence_size = tf.shape(inputs)[1]
      batch_size = tf.shape(inputs)[0]
      stop = tf.range(sequence_size)
      
      stop /= sequence_size
      
      stop = tf.expand_dims(stop,
                            axis = 0)
      stop = tf.tile(stop,
                     [batch_size, 1])
      stop = tf.cast(tf.expand_dims(stop,
                                    axis = 2),
                     dtype = tf.float32)
      return tf.concat([inputs, stop],
                       axis = -1)
    elif self.arg.pos == 'tanh_stop':
      sequence_size = tf.shape(inputs)[1]
      batch_size = tf.shape(inputs)[0]
      stop = tf.range(sequence_size)
      stop = tf.cast(stop,
                     dtype = tf.float32)
      sequence_size = tf.cast(sequence_size,
                              dtype = tf.float32)
      
      gamma = 3.0
      stop = tf.nn.tanh(gamma * stop/sequence_size) + 1 - tf.nn.tanh(gamma)
      
      stop = tf.expand_dims(stop,
                            axis = 0)
      stop = tf.tile(stop,
                     [batch_size, 1])
      stop = tf.expand_dims(stop,
                            axis = 2)
      return tf.concat([inputs, stop],
                       axis = -1)
    elif self.arg.pos == 'exp_stop':
      sequence_size = tf.shape(inputs)[1]
      batch_size = tf.shape(inputs)[0]
      stop = tf.range(sequence_size)
      stop = tf.cast(stop,
                     dtype = tf.float32)
      sequence_size = tf.cast(sequence_size,
                              dtype = tf.float32)
      
      gamma = 3.0
      stop = tf.exp(gamma * (stop - sequence_size) / sequence_size)
      
      stop = tf.expand_dims(stop,
                            axis = 0)
      stop = tf.tile(stop,
                     [batch_size, 1])
      stop = tf.expand_dims(stop,
                            axis = 2)
      return tf.concat([inputs, stop],
                       axis = -1)
    else:
      return inputs
    
def argument():
  arg = optimize.argument()
  arg.dropout_type = 'vanilla' # 'vanilla', 'broadcast', 'alpha'
  arg.ffd = 'transformer_ffd' # 'transformer_ffd' 'sru' 'sepconv'
  arg.loss = 'sparse_softmax_cross_entropy_with_logits'
  arg.pos = 'timing' # 'timing' 'emb' 'linear_stop' 'tanh_stop' 'exp_stop'
  
  arg.act_epsilon = 0.001 # a hyperparameter that the ACT mechanism uses
  arg.act_loss_weight = 0.01 # a hyperparameter specifing the auxillary ACT loss in comparison to the total model weight
  arg.filter_size = 1024 # the filter size
  arg.head_size = 64 # the size of each head
  arg.hidden_size = 256 # the hidden size of each model
  arg.input_max_length = 10 # the maximum size of the input sequence size
  arg.input_vocab_size = 1000 # the input vocab size
  arg.label_smoothing = 1.0 # the label smoothing hyperparameter
  arg.max_encoder_steps = 8 # the maximum number of encoder layers
  arg.max_decoder_steps = 8 # the maximum number of decoder layers
  arg.max_relative_position = 100 # used for relative attention
  arg.num_heads = 8 # the number of heads in a self-attention mechanism
  arg.target_max_length = 10 # the maximum size of the target sequence size
  arg.target_vocab_size = 1000 # the target vocab size
  arg.weight_decay_hyperparameter = 0.001 # the weight decay hyperparameter
  
  arg.classification = False # whether the final output is a sequence, or single label
  arg.deparameterize = False # KEEP AS FALSE
  arg.mask_loss = True # whether parts of the loss is masked
  arg.relative_attention = False # whether to use relative attention
  arg.unidirectional_decoder = True # whether the decoder is unidirectional
  arg.unidirectional_encoder = False # whether the encoder is unidirectional
  arg.use_act = False # whether the Universal Transformer uses an ACT mechanism
  arg.use_decoder = True # whether to use the decoder
  arg.use_mos = False # whether to use an MoS
  arg.use_relu = True # whether the activation functions are ReLU or GELU
  arg.weight_decay_regularization = False # whether to use weight decay
  return arg

if __name__ == '__main__':
  arg = argument()
  
  model = Universal_Transformer(arg)