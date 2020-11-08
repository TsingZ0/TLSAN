import os
import time
import pickle
import numpy as np
import tensorflow as tf

from input import DataInput
from model import Model

max_epochs = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 32

time_line = []
auc_value = []

with open('dataset.pkl', 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(
    config=tf.ConfigProto(gpu_options=gpu_options)
    ) as sess:

  model = Model(user_count, item_count, cate_count, cate_list)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  best_auc = 0.0
  best_prec = [0, 0, 0, 0, 0, 0]
  best_recall = [0, 0, 0, 0, 0, 0]

  lr = 1.0
  start_time = time.time()
  for epoch in range(max_epochs):

    if epoch % 100 == 0 and epoch != 0:
      lr *= 0.5

    epoch_size = train_set.shape[0] // train_batch_size
    loss_sum = 0.0
    for _, uij in DataInput(train_set, train_batch_size):
      loss = model.train(sess, uij, lr)
      loss_sum += loss

    epoch += 1
    print('epoch: %d\ttrain_loss: %.2f\tlr: %.2f' %
          (epoch, loss_sum / epoch_size, lr), end='\t')

    test_auc = model.eval_auc(sess, test_set)
    
    time_line.append(time.time()-start_time)
    auc_value.append(test_auc)

    print('test_auc: %.4f' % test_auc, flush=True)

    prec = model.eval_prec(sess, test_set)
    recall = model.eval_recall(sess, test_set)

    print('Precision:')
    for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
      print('@' + str(k) + ' = %.4f' % prec[i], end=' ')
    print()
    print('Recall:')
    for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
      print('@' + str(k) + ' = %.4f' % recall[i], end=' ')
    print()

    for i in range(6):
      if prec[i] > best_prec[i]:
        best_prec[i] = prec[i]
      if recall[i] > best_recall[i]:
        best_recall[i] = recall[i]
    if best_auc < test_auc:
      best_auc = test_auc
  model.save(sess, 'save_path/bpr')

  print('best test_auc:', best_auc)
  print('Best precision:')
  for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
    print('@' + str(k) + ' = %.4f' % best_prec[i], end=' ')
  print()
  print('Best recall:')
  for i, k in zip(range(6), [1, 10, 20, 30, 40, 50]):
    print('@' + str(k) + ' = %.4f' % best_recall[i], end=' ')
  print()
  print('Finished', flush=True)
  
with open('training_time.pkl', 'wb') as f:
  pickle.dump((time_line, auc_value), f, pickle.HIGHEST_PROTOCOL)
