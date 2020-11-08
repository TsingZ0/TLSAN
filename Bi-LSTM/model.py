import tensorflow as tf

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

class Model(object):

  def __init__(self, user_count, item_count, cate_count, cate_list):

    self.u = tf.placeholder(tf.int32, [None,]) # [B]
    self.i = tf.placeholder(tf.int32, [None,]) # [B]
    self.j = tf.placeholder(tf.int32, [None,]) # [B]
    self.y = tf.placeholder(tf.float32, [None,]) # [B]
    self.hist_i = tf.placeholder(tf.int32, [None, None]) # [B, T]
    self.sl = tf.placeholder(tf.int32, [None,]) # [B]
    self.lr = tf.placeholder(tf.float64, [])


    hidden_units = 64

    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
    item_b = tf.get_variable("item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

    u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)

    ic = tf.gather(cate_list, self.i)
    i_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(cate_emb_w, ic),
        ], 1)
    i_b = tf.gather(item_b, self.i)

    jc = tf.gather(cate_list, self.j)
    j_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(cate_emb_w, jc),
        ], 1)
    j_b = tf.gather(item_b, self.j)

    hc = tf.gather(cate_list, self.hist_i)
    h_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.hist_i),
        tf.nn.embedding_lookup(cate_emb_w, hc),
        ], 2)

    '''
    # uni-directional rnn
    # rnn_output, _ = tf.nn.dynamic_rnn(
    #     build_cell(hidden_units), h_emb, self.sl, dtype=tf.float32)

    hist = extract_axis_1(rnn_output, self.sl-1)
    # u_emb = tf.concat([u_emb, hist], axis=1)
    '''

    cell_fw = build_cell(hidden_units)
    cell_bw = build_cell(hidden_units)
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, h_emb, self.sl, dtype=tf.float32)
    hist = tf.concat([
        extract_axis_1(rnn_output[0], self.sl-1),
        tf.reshape(rnn_output[1][:, 0, :], [-1, hidden_units]),
        ], axis=1)
    hist = tf.layers.dense(hist, hidden_units)

    u_emb = hist

    # MF predict: u_i > u_j
    x = i_b - j_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1) # [B]
    self.logits = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1)
    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))

    # logits for all item:
    all_emb = tf.concat([
        item_emb_w,
        tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
    self.eval_logits = tf.matmul(u_emb, all_emb, transpose_b=True) + item_b
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
        tf.nn.l2_loss(user_emb_w),
        tf.nn.l2_loss(item_emb_w),
        tf.nn.l2_loss(cate_emb_w),
        ])

    regulation_rate = 0.00005
    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        ) + regulation_rate * l2_norm

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)


  def train(self, sess, uij, l):
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        })
    return loss

  def eval_auc(self, sess, uij):
    u_auc = sess.run(self.mf_auc, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        })
    return u_auc

  def eval_prec(self, sess, uij):
    prec_update_ops = [self.prec_update_1, self.prec_update_10, 
        self.prec_update_20, self.prec_update_30, 
        self.prec_update_40, self.prec_update_50]

    return sess.run(prec_update_ops, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
    })

  def eval_recall(self, sess, uij):
    recall_update_ops = [self.recall_update_1, self.recall_update_10, 
        self.recall_update_20, self.recall_update_30, 
        self.recall_update_40, self.recall_update_50]

    return sess.run(recall_update_ops, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
    })

  def test(self, sess, uid, hist_i, sl):
    return sess.run(self.logits_all, feed_dict={
        self.u: uid,
        self.hist_i: hist_i,
        self.sl: sl,
        })

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

def build_single_cell(hidden_units):
  cell_type = LSTMCell
  # cell_type = GRUCell
  cell = cell_type(hidden_units)
  return cell

def build_cell(hidden_units, depth=1):
  cell_list = [build_single_cell(hidden_units) for i in range(depth)]
  return MultiRNNCell(cell_list)
