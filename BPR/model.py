import tensorflow as tf

class Model(object):

  def __init__(self, user_count, item_count, cate_count, cate_list):
    self.u = tf.placeholder(tf.int32, [None,])
    self.i = tf.placeholder(tf.int32, [None,])
    self.j = tf.placeholder(tf.int32, [None,])
    self.lr = tf.placeholder(tf.float64, [])

    user_emb_w = tf.get_variable("user_emb_w", [user_count, 64])
    item_emb_w = tf.get_variable("item_emb_w", [item_count, 32])
    item_b = tf.get_variable("item_b", [item_count])
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, 32])
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

    # MF predict: u_i > u_j
    x = i_b - j_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1)
    self.logits = tf.sigmoid(x)

    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    # average AUC = mean( auc for each user in test set)
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

    l2_norm = tf.add_n([
        tf.nn.l2_loss(u_emb),
        tf.nn.l2_loss(i_emb),
        tf.nn.l2_loss(j_emb),
        ])

    reg_rate = 5e-5
    self.bprloss = reg_rate * l2_norm - tf.reduce_mean(tf.log(self.logits))

    opt = tf.train.GradientDescentOptimizer
    self.train_op = opt(self.lr).minimize(self.bprloss)

  def train(self, sess, uij, l):
    loss, _ = sess.run([self.bprloss, self.train_op], feed_dict={
        self.u: uij[:, 0],
        self.i: uij[:, 1],
        self.j: uij[:, 2],
        self.lr: l,
        })
    return loss

  def eval_auc(self, sess, test_set):
    return sess.run(self.mf_auc, feed_dict={
        self.u: test_set[:, 0],
        self.i: test_set[:, 1],
        self.j: test_set[:, 2],
        })

  def eval_prec(self, sess, test_set):
    prec_update_ops = [self.prec_update_1, self.prec_update_10, 
        self.prec_update_20, self.prec_update_30, 
        self.prec_update_40, self.prec_update_50]

    return sess.run(prec_update_ops, feed_dict={
        self.u: test_set[:, 0],
        self.i: test_set[:, 1],
        self.j: test_set[:, 2],
    })

  def eval_recall(self, sess, test_set):
    recall_update_ops = [self.recall_update_1, self.recall_update_10, 
        self.recall_update_20, self.recall_update_30, 
        self.recall_update_40, self.recall_update_50]

    return sess.run(recall_update_ops, feed_dict={
        self.u: test_set[:, 0],
        self.i: test_set[:, 1],
        self.j: test_set[:, 2],
    })
  
  def test(self, sess, uid):
    return sess.run(self.logits_all, feed_dict={
        self.u: uid,
        })

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
