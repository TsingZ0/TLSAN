import os
import json
import numpy as np
import tensorflow as tf

class Model(object):
  def __init__(self, config):
    self.config = config

    # Summary Writer
    self.train_writer = tf.summary.FileWriter(config['model_dir'] + '/train')
    self.eval_writer = tf.summary.FileWriter(config['model_dir'] + '/eval')

    # Building network
    self.init_placeholders()
    self.build_model()
    self.init_optimizer()


  def init_placeholders(self):
    # [B] item id
    self.i = tf.placeholder(tf.int32, [None,]) #用于AUC

    # [B] item label
    self.y = tf.placeholder(tf.float32, [None,])

    # [B, T] user's history item id
    self.hist_i = tf.placeholder(tf.int32, [None, None])

    # [B] valid length of `hist_i`
    self.sl = tf.placeholder(tf.int32, [None,])

    # learning rate
    self.lr = tf.placeholder(tf.float64, [])

    # whether it's training or not
    self.is_training = tf.placeholder(tf.bool, [])


  def build_model(self):
    item_emb_w = tf.get_variable(
        "item_emb_w",
        [self.config['item_count'], self.config['embedding_size']])
    position_w = tf.get_variable(
        "weights_position",
        [self.config['kernel_size'], self.config['max_len'], self.config['embedding_size']])
    linear_w = tf.get_variable(
        "weights_bilinear",
        [self.config['embedding_size'], self.config['embedding_size']])

    i_emb = tf.nn.embedding_lookup(item_emb_w, self.i)
    h_emb = tf.nn.embedding_lookup(item_emb_w, self.hist_i)

    num_step = tf.shape(h_emb)[1]
    dropout_rate = self.config['dropout']
    kernel_size = self.config['kernel_size']
    num_units = h_emb.get_shape().as_list()[-1]

    u_emb = PACA(
        h_emb, 
        self.sl, 
        position_w, 
        linear_w, 
        num_step,  
        num_units, 
        kernel_size, 
        dropout_rate, 
        self.is_training, 
        False)

    self.logits = tf.reduce_sum(tf.multiply(u_emb, i_emb), 1)

    # ============== Eval ===============
    self.eval_logits = tf.matmul(u_emb, item_emb_w, transpose_b=True)
    self.i64 = tf.cast(self.i, tf.int64)
    with tf.variable_scope("metric"):
      # precision_at_k
      self.prec_1, self.prec_update_1 = tf.metrics.precision_at_k(labels=self.i64, predictions=self.eval_logits, k=1)
      self.prec_10, self.prec_update_10 = tf.metrics.precision_at_k(labels=self.i64, predictions=self.eval_logits, k=10)
      self.prec_20, self.prec_update_20 = tf.metrics.precision_at_k(labels=self.i64, predictions=self.eval_logits, k=20)
      self.prec_30, self.prec_update_30 = tf.metrics.precision_at_k(labels=self.i64, predictions=self.eval_logits, k=30)
      self.prec_40, self.prec_update_40 = tf.metrics.precision_at_k(labels=self.i64, predictions=self.eval_logits, k=40)
      self.prec_50, self.prec_update_50 = tf.metrics.precision_at_k(labels=self.i64, predictions=self.eval_logits, k=50)
      # recall_at_k
      self.recall_1, self.recall_update_1 = tf.metrics.recall_at_k(labels=self.i64, predictions=self.eval_logits, k=1)
      self.recall_10, self.recall_update_10 = tf.metrics.recall_at_k(labels=self.i64, predictions=self.eval_logits, k=10)
      self.recall_20, self.recall_update_20 = tf.metrics.recall_at_k(labels=self.i64, predictions=self.eval_logits, k=20)
      self.recall_30, self.recall_update_30 = tf.metrics.recall_at_k(labels=self.i64, predictions=self.eval_logits, k=30)
      self.recall_40, self.recall_update_40 = tf.metrics.recall_at_k(labels=self.i64, predictions=self.eval_logits, k=40)
      self.recall_50, self.recall_update_50 = tf.metrics.recall_at_k(labels=self.i64, predictions=self.eval_logits, k=50)

    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    # Loss
    l2_norm = tf.add_n([
        tf.nn.l2_loss(item_emb_w),
        tf.nn.l2_loss(position_w),
        ])

    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        ) + self.config['regulation_rate'] * l2_norm

    self.train_summary = tf.summary.merge([
        tf.summary.histogram('embedding/1_item_emb', item_emb_w),
        tf.summary.histogram('embedding/2_position_weight', position_w),
        tf.summary.histogram('attention_output', u_emb),
        tf.summary.scalar('L2_norm Loss', l2_norm),
        tf.summary.scalar('Training Loss', self.loss),
        ])


  def init_optimizer(self):
    # Gradients and SGD update operation for training the model
    trainable_params = tf.trainable_variables()
    if self.config['optimizer'] == 'adadelta':
      self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
    elif self.config['optimizer'] == 'adam':
      self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
    elif self.config['optimizer'] == 'rmsprop':
      self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
    else:
      self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

    # Compute gradients of loss w.r.t. all trainable variables
    gradients = tf.gradients(self.loss, trainable_params)

    # Clip gradients by a given maximum_gradient_norm
    clip_gradients, _ = tf.clip_by_global_norm(
        gradients, self.config['max_gradient_norm'])

    # Update the model
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)



  def train(self, sess, uij, l, add_summary=False):

    input_feed = {
        self.i: uij[0],
        self.y: uij[1],
        self.hist_i: uij[2],
        self.sl: uij[3],
        self.lr: l,
        self.is_training: True,
        }

    output_feed = [self.loss, self.train_op]

    if add_summary:
      output_feed.append(self.train_summary)

    outputs = sess.run(output_feed, input_feed)

    if add_summary:
      self.train_writer.add_summary(
          outputs[2], global_step=self.global_step.eval())

    return outputs[0]

  def eval_auc(self, sess, uij):
    res1 = sess.run(self.logits, feed_dict={
        self.i: uij[0],
        self.hist_i: uij[2],
        self.sl: uij[3],
        self.is_training: False,
        })
    res2 = sess.run(self.logits, feed_dict={
        self.i: uij[1],
        self.hist_i: uij[2],
        self.sl: uij[3],
        self.is_training: False,
        })
    return np.mean(res1 - res2 > 0)

  def eval_prec(self, sess, uij):
    prec_update_ops = [self.prec_update_1, self.prec_update_10, 
        self.prec_update_20, self.prec_update_30, 
        self.prec_update_40, self.prec_update_50]
    
    return sess.run(prec_update_ops, feed_dict={
        self.i: uij[0],
        self.hist_i: uij[2],
        self.sl: uij[3],
        self.is_training: False,
    })

  def eval_recall(self, sess, uij):
    recall_update_ops = [self.recall_update_1, self.recall_update_10, 
        self.recall_update_20, self.recall_update_30, 
        self.recall_update_40, self.recall_update_50]
    
    return sess.run(recall_update_ops, feed_dict={
        self.i: uij[0],
        self.hist_i: uij[2],
        self.sl: uij[3],
        self.is_training: False,
    })

    
  def save(self, sess):
    checkpoint_path = os.path.join(self.config['model_dir'], 'PACA')
    saver = tf.train.Saver()
    save_path = saver.save(
        sess, save_path=checkpoint_path, global_step=self.global_step.eval())
    json.dump(self.config,
              open('%s-%d.json' % (checkpoint_path, self.global_step.eval()), 'w'),
              indent=2)
    print('model saved at %s' % save_path, flush=True)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
    print('model restored from %s' % path, flush=True)


def PACA(
      session_emb, 
      sl, 
      position_w, 
      linear_w, 
      num_step, 
      num_units, 
      kernel_size, 
      dropout_rate, 
      is_training, 
      reuse):
  with tf.variable_scope("all", reuse=reuse):
    paa_h = PositionAwareAttention(
      session_emb, 
      sl, 
      position_w, 
      num_step, 
      num_units, 
      kernel_size, 
      dropout_rate, 
      is_training, 
      "PAA",
      reuse)
    
    final_state = Bilinear(
      paa_h, 
      linear_w, 
      dropout_rate, 
      is_training, 
      "Linear",
      reuse)

    return final_state


def PositionAwareAttention(
      session_emb, 
      sl, 
      position_w, 
      num_step, 
      num_units, 
      kernel_size, 
      dropout_rate, 
      is_training, 
      scope, 
      reuse):
  with tf.variable_scope(scope, reuse=reuse):
    session_emb = tf.layers.dropout(session_emb, 
        rate=dropout_rate, training=is_training)
    session_emb = tf.transpose(session_emb, [1, 0, 2])    

    # Masking
    mask = tf.sequence_mask(sl, tf.shape(session_emb)[0]) #bat*sq
    mask = tf.transpose(mask)
    ones = tf.ones_like(mask, dtype=tf.float32)
    zeros = tf.zeros_like(mask, dtype=tf.float32)
    mask = tf.where(mask, ones, zeros)

    session_emb = tf.multiply(session_emb, tf.expand_dims(mask,axis=2))
    tmp_emb = tf.sigmoid(session_emb)

    kernel_emb =[]
    for i in range(kernel_size):
      tmp_wp = position_w[i,:num_step, :]
      tmp_wp = tf.expand_dims(tmp_wp, axis=2)
      sim_matrix = tf.matmul(tmp_emb,tmp_wp) #sp*bat*1
      sim_matrix  = tf.reduce_sum(sim_matrix,axis=2)#sp*bat
      kernel_emb.append(sim_matrix)

    sim_matrix = tf.stack(kernel_emb)
    sim_matrix = tf.reduce_max(sim_matrix,axis=0)

    tmp = tf.multiply(sim_matrix,mask) #step*bat
    tmp = tf.nn.softmax(tmp,axis=0)
    att = tf.multiply(tmp,mask)
    p = tf.reduce_sum(att,axis=0,keepdims=True)
    att_alpha = tf.div(att,p) # step*bat
    paa_matrix = tf.multiply(session_emb,tf.expand_dims(att_alpha,axis=2))
    paa_h  = tf.reduce_sum(paa_matrix,axis=0) #bat*dimte

    return paa_h

def Bilinear(
      paa_h, 
      linear_w, 
      dropout_rate, 
      is_training, 
      scope, 
      reuse):
    with tf.variable_scope(scope, reuse=reuse):
      paa_h = tf.layers.dropout(paa_h, 
          rate=dropout_rate, training=is_training)

      final_state = tf.matmul(paa_h, linear_w)

      return final_state