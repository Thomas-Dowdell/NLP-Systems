import numpy as np
import tensorflow as tf

class Neural_GPU():
  def __init__(self):
    self.batch_size = 128
    self.sequence_size = 5
    self.embed_dim = 24
    self.vocab_size = 20
    self.width = 3
    self.kernel_width = 3
    self.kernel_height = 1
    self.layer = 2
    self.learning_rate = 1e-4
    self.clip_norm = None
    self.stddev = None
    self.inputs = tf.placeholder(tf.int32,
                                 shape = [self.batch_size, self.sequence_size])
    self.targets = tf.placeholder(tf.int32,
                                  shape = [self.batch_size, self.sequence_size])
    self.dropout_rate = tf.placeholder(tf.float32)
    self.embed_weight = tf.get_variable('embed_weight',
                                        shape = [self.vocab_size, self.embed_dim])
    s0 = tf.nn.embedding_lookup(self.embed_weight,
                                self.inputs)
    s0 = tf.reshape(s0,
                    shape = [self.batch_size, 1, self.sequence_size, self.embed_dim])
    s0 = tf.concat([s0, tf.zeros([self.batch_size, self.width - 1, self.sequence_size, self.embed_dim])],
                   axis = 1)
    sfin = self.neural_gpu(s0)
    output = sfin[:,0,:,:]
    self.output_weight = tf.get_variable('output_weight',
                                         shape = [self.embed_dim, self.vocab_size])
    self.output_bias = tf.get_variable('output_bias',
                                       shape = [self.vocab_size])
    self.output = tf.tensordot(output,
                               self.output_weight,
                               axes = 1) + self.output_bias
    self.predict = tf.argmax(self.output,
                             axis = -1,
                             output_type = tf.int32)
    self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.one_hot(self.targets,
                                                                               depth = self.vocab_size),
                                                           logits = self.output,
                                                           axis = -1)
    self.loss = tf.reduce_mean(self.loss)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    if self.clip_norm is not None:
      grads_and_vars = [(tf.clip_by_norm(g, 
                                       self.clip_norm), 
                       v) for g, v in grads_and_vars]
    if self.stddev is not None:
      grads_and_vars = [(self.add_gradient_noise(g, 
                                                stddev = self.stddev), 
                         v) for g, v in grads_and_vars]
    self.train_op = optimizer.apply_gradients(grads_and_vars)
    correct_prediction = tf.equal(self.predict,
                                  self.targets)
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                           tf.float32))
    
  def add_gradient_noise(self, gradient,
                         stddev):
    if gradient is None:
      return None
    return gradient + tf.random_normal(tf.shape(gradient),
                                       stddev = stddev)
    
  def neural_gpu(self, s):
    def neural_gpu(i, s):
      for l in range(self.layer):
        with tf.variable_scope('layer_{}'.format(l + 1)):
          s = self.cgru(s)
      s = tf.nn.dropout(s,
                        rate = self.dropout_rate)
      return i + 1, s
    i = 0
    cond = lambda i, s: tf.less(i, self.sequence_size)
    body = lambda i, s: neural_gpu(i, s)
    i, s = tf.while_loop(cond = cond,
                         body = body,
                         loop_vars = [i, s])
    return s
    
  def update_gate(self, s):
    with tf.variable_scope('update_gate'):
      weights = tf.get_variable('weight',
                                shape = [self.kernel_width, self.kernel_height, self.embed_dim, self.embed_dim])
      bias = tf.get_variable('bias',
                             shape = [self.embed_dim])
      return tf.nn.convolution(input = s,
                               filter = weights,
                               padding = 'SAME') + bias
    
  def reset_gate(self, s):
    with tf.variable_scope('reset_gate'):
      weights = tf.get_variable('weight',
                                shape = [self.kernel_width, self.kernel_height, self.embed_dim, self.embed_dim])
      bias = tf.get_variable('bias',
                             shape = [self.embed_dim])
      return tf.nn.convolution(input = s,
                               filter = weights,
                               padding = 'SAME') + bias
    
  def cgru(self, s):
    weights = tf.get_variable('weight',
                              shape = [self.kernel_width, self.kernel_height, self.embed_dim, self.embed_dim])
    bias = tf.get_variable('bias',
                           shape = [self.embed_dim])
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
  
if __name__ == '__main__':
  import sys
  sys.path.append('Datasets')
  from tasks import copy # any task could be analyzed
  
  epochs = 5
  iterations = 100
  
  model = Neural_GPU()
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  for epoch in range(1, epochs + 1):
    train_cost = []
    for iteration in range(iterations):
      trainX, trainY = copy(model.batch_size,
                            model.sequence_size,
                            model.vocab_size)
      _, cost = sess.run([model.train_op, model.loss],
                         feed_dict = {model.inputs: trainX,
                                      model.targets: trainY,
                                      model.dropout_rate: 0.06})
      assert not np.isnan(cost), 'Epoch {}, Iteration {}'.format(epoch,
                                                                 iteration + 1)
      train_cost.append(cost)
    print('Epoch {}, Average Training Loss {:.4f}, Final Training Loss {:.4f}'.format(epoch,
                                                                                      np.mean(train_cost),
                                                                                      cost))
    testX, testY = copy(model.batch_size,
                        model.sequence_size,
                        model.vocab_size)
    test_accuracy = sess.run(model.accuracy,
                             feed_dict = {model.inputs: testX,
                                          model.targets: testY,
                                          model.dropout_rate: 0.0})
    print('Accuracy {:.4f}'.format(test_accuracy))
  sess.close()