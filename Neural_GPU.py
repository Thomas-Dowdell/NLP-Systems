import numpy as np
import tensorflow as tf
  
if __name__ == '__main__':
  import sys
  sys.path.append('Datasets')
  from tasks import copy # any task could be analyzed
  sys.path.append('Utils')
  from Utils import Neural_GPU
  
  epochs = 5
  iterations = 100
  
  model = Neural_GPU.Neural_GPU()
  
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
                                      model.training: True,
                                      model.learning_rate: 0.01,
                                      model.keep_prob: 0.94})
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
                                          model.training: False,
                                          model.keep_prob: 1.0})
    print('Accuracy {:.4f}'.format(test_accuracy))
  sess.close()
