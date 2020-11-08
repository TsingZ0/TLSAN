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
    # [B] user id
    self.u = tf.placeholder(tf.int32, [None,])

    # [B] item id
    self.i = tf.placeholder(tf.int32, [None,])
    self.j = tf.placeholder(tf.int32, [None,])
    self.s = tf.placeholder(tf.int32, [None, None])

    # [B] item label
    self.y = tf.placeholder(tf.float32, [None,])

    # learning rate
    self.lr = tf.placeholder(tf.float64, [])


  def build_model(self):
    item_emb_w = tf.get_variable(
        "item_emb_w",
        [self.config['item_count'], self.config['embedding_size']])
    short_w = tf.get_variable(
        "short_w",
        [self.config['item_count'], self.config['embedding_size']])
    long_w = tf.get_variable(
        "long_w",
        [self.config['user_count'], self.config['embedding_size']])
    D = [1.0 / (x+1) for x in range(self.config['k'])]
    D.reverse()
    D = tf.convert_to_tensor(D)
    D = tf.tile(tf.expand_dims(D, -1), [1, self.config['embedding_size']])

    is_emb = tf.nn.embedding_lookup(short_w, self.s)
    hi_emb = tf.nn.embedding_lookup(item_emb_w, self.i)
    hj_emb = tf.nn.embedding_lookup(item_emb_w, self.j)
    u_emb = tf.nn.embedding_lookup(long_w, self.u)

    s_emb = tf.reduce_sum(tf.multiply(is_emb, D), 1)
    p = u_emb + tf.multiply(self.config['alpha'], s_emb) 

    self.r_i = tf.reduce_sum(tf.multiply(p, hi_emb), 1)
    self.r_j = tf.reduce_sum(tf.multiply(p, hj_emb), 1)

    self.x = self.r_i - self.r_j

    # ============== Eval ===============
    all_emb = item_emb_w
    self.eval_logits = tf.matmul(p, all_emb, transpose_b=True)
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
        tf.nn.l2_loss(u_emb),
        tf.nn.l2_loss(is_emb),
        tf.nn.l2_loss(hi_emb),
        tf.nn.l2_loss(hj_emb),
        ])

    self.loss = tf.reduce_sum(
        -tf.math.log(tf.clip_by_value(tf.math.sigmoid(self.x), 1e-8, 1.0))
        ) + self.config['regulation_rate'] * l2_norm

    # self.train_summary = tf.summary.merge([
    #     tf.summary.histogram('embedding/1_item_emb', item_emb_w),
    #     tf.summary.histogram('embedding/2_user_emb', long_w),
    #     tf.summary.scalar('L2_norm Loss', l2_norm),
    #     tf.summary.scalar('Training Loss', self.loss),
    #     ])


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
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.s: uij[3],
        self.lr: l, 
        }

    output_feed = [self.loss, self.train_op]

    # if add_summary:
    #   output_feed.append(self.train_summary)

    outputs = sess.run(output_feed, input_feed)

    # if add_summary:
    #   self.train_writer.add_summary(
    #       outputs[2], global_step=self.global_step.eval())

    return outputs[0]

  def eval_auc(self, sess, uij):
    res = sess.run(self.x, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.s: uij[3],
        })
    return np.mean(res > 0)

  def eval_prec(self, sess, uij):
    prec_update_ops = [self.prec_update_1, self.prec_update_10, 
        self.prec_update_20, self.prec_update_30, 
        self.prec_update_40, self.prec_update_50]
    
    return sess.run(prec_update_ops, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.s: uij[3],
        })

  def eval_recall(self, sess, uij):
    recall_update_ops = [self.recall_update_1, self.recall_update_10, 
        self.recall_update_20, self.recall_update_30, 
        self.recall_update_40, self.recall_update_50]
    
    return sess.run(recall_update_ops, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.s: uij[3],
        })

     
  def save(self, sess):
    checkpoint_path = os.path.join(self.config['model_dir'], 'lspm')
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
