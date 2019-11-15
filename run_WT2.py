import time
import numpy as np
import tensorflow as tf

import sys
sys.path.append('Utils')
from optimize import warmup_learning_rate

def build_and_run_model(model,
                        arg):
  def _prepare(data):
    for seq in range(data.shape[1]):
      if np.array_equal(data[:,seq],
                        np.zeros_like(data[:,seq])):
        return data[:,:seq + 1]
    return data

  arg.encoder_layers = 4
  arg.filter_size = 512
  arg.hidden_size = 128
  arg.input_vocab_size = 32001
  arg.target_vocab_size = 32001
  
  arg.mask_loss = True
  arg.relative_attention = False
  arg.unidirectional_encoder = True
  arg.use_decoder = False
  arg.use_relu = True
  
  arg.layers = arg.encoder_layers
  arg.vocab_size = arg.input_vocab_size
  
  model = model(arg)

  print(model.name)
  print('Hidden size: {}'.format(arg.hidden_size))
  print('Filter size: {}'.format(arg.filter_size))
  if model.name == 'Experimental-Transformer':
    print('Attention type: {}'.format(arg.att))
  print('Trainable Variables: {}'.format(np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])))
  print('')
  '''
  try:
    train_data = np.load('../../Dataset/PTB/Word/PTB.train.npy')
    test_data = np.load('../../Dataset/PTB/Word/PTB.test.npy')
    valid_data = np.load('../../Dataset/PTB/Word/PTB.valid.npy')
  except:
    train_data = np.load('PTB.train.npy')
    test_data = np.load('PTB.test.npy')
    valid_data = np.load('PTB.valid.npy')
  train_data = np.reshape(train_data,
                          [52, 809, 83])
  valid_data = np.reshape(valid_data[:3328],
                          [52, 64, -1])
  test_data = np.reshape(test_data[:3744],
                         [52, 72, -1])
  arg.input_vocab_size = arg.target_vocab_size = arg.vocab_size = 10001

  batch_size = 52
  '''
  train_data = np.load('Dataset/WT2/WT2.train.npy')
  test_data = np.concatenate([np.load('Dataset/WT2/WT2.test.npy'), np.zeros([18, 176])], axis = 0)
  valid_data = np.concatenate([np.load('Dataset/WT2/WT2.valid.npy'), np.zeros([70, 243])], axis = 0)
  train_data = np.reshape(train_data,
                          [25, 2961, 275])
  valid_data = np.reshape(valid_data,
                          [25, 316, 243])
  test_data = np.reshape(test_data,
                         [25, 360, 176])
  arg.input_vocab_size = arg.target_vocab_size = arg.vocab_size = 32001
  batch_size = 25
  

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  lr = warmup_learning_rate(arg.hidden_size ** -0.5,
                            warmup = 30000)
  print(time.asctime())

  for epoch in range(25):
    loss_array = []
    acc_array = []
    prev_mems = np.zeros([arg.layers, batch_size, 0, arg.hidden_size])
    for iteration in range(train_data.shape[1]):
      trainX = _prepare(train_data[:,iteration])
      trainX, trainY = trainX[:,:-1], trainX[:,1:]
      feed_dict = {model.inputs: trainX,
                   model.targets: trainY,
                   model.training: True,
                   model.keep_prob: 0.7,
                   model.loss_mask: np.where(trainX == 0,
                                             0.0,
                                             1.0),
                   model.learning_rate: lr()}
      if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
        feed_dict[model.memory] = prev_mems
      _, cost, acc = sess.run([model.train_op, model.cost, model.accuracy],
                              feed_dict = feed_dict)
      if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
        prev_mems = sess.run(model.new_mems,
                             feed_dict = feed_dict)
      loss_array.append(cost)
      acc_array.append(acc)
    x = sess.run(model.predict,
                 feed_dict = feed_dict)
    #print(trainY[0])
    #print(np.multiply(x[0],
    #                  np.where(trainX[0] == 0,
    #                           0.0,
    #                           1.0)))
    test_loss_array = []
    test_acc_array = []
    prev_mems = np.zeros([arg.layers, batch_size, 0, arg.hidden_size])
    for iteration in range(valid_data.shape[1]):
      validX = _prepare(valid_data[:,iteration])
      validX, validY = validX[:,:-1], validX[:,1:]
      feed_dict = {model.inputs: validX,
                   model.targets: validY,
                   model.training: False,
                   model.keep_prob: 1.0,
                   model.loss_mask: np.where(validX == 0,
                                             0.0,
                                             1.0)}
      if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
        feed_dict[model.memory] = prev_mems
      cost, acc, predict = sess.run([model.cost, model.accuracy, model.predict],
                                    feed_dict = feed_dict)
      if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
        prev_mems = sess.run(model.new_mems,
                             feed_dict = feed_dict)
      test_loss_array.append(cost)
      test_acc_array.append(acc)
    print('Epoch {}, Training Loss {:.4f} Valid Loss {:.4f}'.format(epoch + 1,
                                                                   np.mean(loss_array),
                                                                   np.mean(test_loss_array)))
    print('Training Accuracy {:.4f} Valid Accuracy {:.4f}'.format(np.mean(acc_array),
                                                                  np.mean(test_acc_array)))
    x = sess.run(model.predict,
                 feed_dict = feed_dict)
    #print(validY[0])
    #print(np.multiply(x[0],
    #                  np.where(validX[0] == 0,
    #                           0.0, 
    #                           1.0)))
    print(time.asctime())
    print('')
  
  test_loss_array = []
  test_acc_array = []
  prev_mems = np.zeros([arg.layers, batch_size, 0, arg.hidden_size])
  for iteration in range(test_data.shape[1]):
    testX = _prepare(test_data[:,iteration])
    testX, testY = testX[:,:-1], testX[:,1:]
    feed_dict = {model.inputs: testX,
                 model.targets: testY,
                 model.training: False,
                 model.keep_prob: 1.0,
                 model.loss_mask: np.where(testX == 0,
                                           0.0,
                                           1.0)}
    if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
      feed_dict[model.memory] = prev_mems
    cost, acc = sess.run([model.cost, model.accuracy],
                         feed_dict = feed_dict)
    if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
      prev_mems = sess.run(model.new_mems,
                           feed_dict = feed_dict)
    test_loss_array.append(cost)
    test_acc_array.append(acc)
  print('Test Loss {:.4f}, Test Accuracy {:.4f}'.format(np.mean(test_loss_array),
                                                        np.mean(test_acc_array)))
  print(testY[0])
  x = sess.run(model.predict,
               feed_dict = feed_dict)
  print(np.multiply(x[0],
                    np.where(testX[0] == 0,
                             0.0,
                             1.0)))
  sess.close()
  
  del sess
  del model
  del train_data
  del valid_data
  del test_data
  del arg
  del lr
  del prev_mems
  
  return True

if __name__ == '__main__':
  from Transformer import Transformer as model, argument
  arg = argument()
  build_and_run_model(model,
                      arg)
