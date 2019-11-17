import os
import sys
import json
import numpy as np

def save(filename, obj):
	with open(filename, 'w') as file:
		file.write(json.dumps(obj))
		file.close()
		
def load(filename):
	return json.loads(open(filename, 'r').read())

def preprocess_train_data():
  file = 'ptb.train.txt'
  train_data = np.zeros([42068, 82])
  
  word2idx = load('word2idx')
  file = open(file,
              'r')
  txt_file = file.readlines()
  file.close()
  for l in range(len(txt_file)):
    line = txt_file[l].split()
    for w in range(len(line)):
      word = line[w]
      test_data[l,w] = word2idx[word]
  np.save('PTB.test.npy',
          test_data)

def preprocess_valid_data():
  file = 'ptb.valid.txt'
  valid_data = np.zeros([3370, 74])
  
  word2idx = load('word2idx')
  file = open(file,
              'r')
  txt_file = file.readlines()
  file.close()
  for l in range(len(txt_file)):
    line = txt_file[l].split()
    for w in range(len(line)):
      word = line[w]
      test_data[l,w] = word2idx[word]
  np.save('PTB.test.npy',
          test_data)

def preprocess_test_data():
  file = 'ptb.test.txt'
  test_data = np.zeros([3761, 77])
  
  word2idx = load('word2idx')
  file = open(file,
              'r')
  txt_file = file.readlines()
  file.close()
  for l in range(len(txt_file)):
    line = txt_file[l].split()
    for w in range(len(line)):
      word = line[w]
      test_data[l,w] = word2idx[word]
  np.save('PTB.test.npy',
          test_data)
    
if __name__ == '__main__':
  preprocess_train_data()
  preprocess_valid_data()
  preprocess_test_data()