import os
import json
import numpy as np
import tensorflow as tf
from functools import reduce
from operator import mul

max_length = 90

VERY_BIG_NUMBER = 1e30
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

class Model(object):
    def __init__(self, config, item_cate_list, user_cate_list):
        self.config = config

        # Summary Writer
        self.train_writer = tf.summary.FileWriter(config['model_dir'] + '/train')
        self.eval_writer = tf.summary.FileWriter(config['model_dir'] + '/eval')

        # Building network
        self.init_placeholders()
        self.build_model(item_cate_list, user_cate_list)
        self.init_optimizer()


    def init_placeholders(self):
        # [bs] user id
        self.u = tf.placeholder(tf.int32, [None,])

        # [bs] item id
        self.i = tf.placeholder(tf.int32, [None,])

        # [bs] item label
        self.y = tf.placeholder(tf.float32, [None,])

        # [bs, sl] history item id
        self.hist_i = tf.placeholder(tf.int32, [None, None])
        self.hist_i_new = tf.placeholder(tf.int32, [None, None])
        
        # [bs, sl] history item purchase time
        self.hist_t = tf.placeholder(tf.int32, [None, None])

        # [bs] valid length of `hist_i`
        self.sl = tf.placeholder(tf.int32, [None,])
        self.sl_new = tf.placeholder(tf.int32, [None,])

        # learning rate
        self.lr = tf.placeholder(tf.float32, [])

        # training or not
        self.is_training = tf.placeholder(tf.bool, [])


    def build_model(self, item_cate_list, user_cate_list):
        # parameter for position matrix
        gamma = tf.get_variable(
            "gamma_parameter", [], 
            initializer=tf.constant_initializer(1.0))
        # item ID embedding
        item_emb = tf.get_variable(
            "item_emb", 
            [self.config['item_count'], self.config['itemid_embedding_size']])
        item_b = tf.get_variable(
            "item_b",
            [self.config['item_count'],],
            initializer=tf.constant_initializer(0.0))
        # user ID embedding
        user_emb = tf.get_variable(
            "user_emb", 
            [self.config['user_count'], self.config['userid_embedding_size']])
        # category embedding
        cate_emb = tf.get_variable(
            "cate_emb",
            [self.config['cate_count'], self.config['cateid_embedding_size']])

        # item embedding
        i_emb = tf.nn.embedding_lookup(item_emb, self.i)
        i_cate_emb = tf.nn.embedding_lookup(cate_emb, tf.gather(item_cate_list, self.i))
        i_emb = tf.concat([i_emb, i_cate_emb], -1)
        i_b = tf.gather(item_b, self.i)
        # all items embedding
        all_cate_emb = tf.nn.embedding_lookup(cate_emb, item_cate_list)
        all_emb = tf.concat([item_emb, all_cate_emb], -1)

        # user embedding
        u_emb = tf.nn.embedding_lookup(user_emb, self.u)
        u_cate_emb = tf.nn.embedding_lookup(cate_emb, tf.gather(user_cate_list, self.u))
        u_emb = tf.concat([u_emb, u_cate_emb], -1)

        # long-term sequential behavior embedding
        h_emb = tf.nn.embedding_lookup(item_emb, self.hist_i)
        h_cate_emb = tf.nn.embedding_lookup(cate_emb, tf.gather(item_cate_list, self.hist_i))        
        h_emb = tf.concat([h_emb, h_cate_emb], -1)
        # short-term sequential behavior embedding
        h_emb_new = tf.nn.embedding_lookup(item_emb, self.hist_i_new)
        h_cate_emb_new = tf.nn.embedding_lookup(cate_emb, tf.gather(item_cate_list, self.hist_i_new))
        h_emb_new = tf.concat([h_emb_new, h_cate_emb_new], -1)

        # directly concatenate position embedding or not
        if self.config['concat_position_emb'] == True:
            t_emb = tf.one_hot(self.hist_t, 12, dtype=tf.float32)
            h_emb = tf.concat([h_emb, t_emb], -1)
            h_emb = tf.layers.dense(h_emb, self.config['hidden_units'])
        

        num_blocks = self.config['num_blocks']
        num_heads = self.config['num_heads']
        dropout_rate = self.config['dropout']
        num_units = h_emb.get_shape().as_list()[-1]


        u_t, self.att0, self.att1 = attention_net(
            user=u_emb,
            enc=h_emb, 
            enc_new=h_emb_new, 
            sl=self.sl,
            sl_new=self.sl_new,
            rep_time=self.hist_t, 
            gamma=gamma, 
            num_units=num_units, 
            num_heads=num_heads, 
            num_blocks=num_blocks, 
            dropout_rate=dropout_rate, 
            is_training=self.is_training, 
            reuse=False)

        self.logits = tf.reduce_sum(tf.multiply(u_t, i_emb), -1) + i_b

        # Eval
        self.eval_logits = tf.matmul(u_t, all_emb, transpose_b=True) + item_b
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
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        # loss
        l2_norm = tf.add_n([
            tf.nn.l2_loss(user_emb),
            tf.nn.l2_loss(item_emb),
            tf.nn.l2_loss(cate_emb),
        ])
        
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))\
             + self.config['regulation_rate'] * l2_norm

        self.train_summary = tf.summary.merge([
            tf.summary.histogram('embedding/0_gamma', gamma),
            tf.summary.histogram('embedding/1_item_emb', item_emb),
            tf.summary.histogram('embedding/2_user_emb', user_emb),
            tf.summary.histogram('embedding/3_history_emb', h_emb),
            tf.summary.histogram('embedding/4_history_emb_new', h_emb_new),
            tf.summary.histogram('attention_output', u_t),
            tf.summary.scalar('L2_norm_user_item Loss', l2_norm),
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
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config['max_gradient_norm'])

        # Update the model
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)
        
    
    def train(self, sess, batch, lr, add_summary=False):

        input_feed = {
            self.u: batch[0],
            self.i: batch[1],
            self.y: batch[2],
            self.hist_i: batch[3],
            self.hist_i_new: batch[4],
            self.hist_t: batch[5],
            self.sl: batch[6],
            self.sl_new: batch[7],
            self.lr: lr,
            self.is_training: True,
        }

        output_feed = [self.loss, self.train_op]

        if add_summary:
            output_feed.append(self.train_summary)

        outputs = sess.run(output_feed, input_feed)

        if add_summary:
            self.train_writer.add_summary(outputs[2], global_step=self.global_step.eval())

        return outputs[0]


    def eval_auc(self, sess, batch):
        #positive_item_list
        res1 = sess.run(self.logits, feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[3],
            self.hist_i_new: batch[4],
            self.hist_t: batch[5],
            self.sl: batch[6],
            self.sl_new: batch[7], 
            self.is_training: False,
        })
        #negative_item_list
        res2 = sess.run(self.logits, feed_dict={
            self.u: batch[0],
            self.i: batch[2],
            self.hist_i: batch[3],
            self.hist_i_new: batch[4],
            self.hist_t: batch[5],
            self.sl: batch[6],
            self.sl_new: batch[7],
            self.is_training: False, 
        })
        
        return np.mean(res1 - res2 > 0)

    def eval_prec(self, sess, batch):

        prec_update_ops = [self.prec_update_1, self.prec_update_10, 
            self.prec_update_20, self.prec_update_30, 
            self.prec_update_40, self.prec_update_50]

        return sess.run(prec_update_ops, feed_dict={ 
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[3],
            self.hist_i_new: batch[4],
            self.hist_t: batch[5],
            self.sl: batch[6],
            self.sl_new: batch[7], 
            self.is_training: False,       
        })

    def eval_recall(self, sess, batch):

        recall_update_ops = [self.recall_update_1, self.recall_update_10, 
            self.recall_update_20, self.recall_update_30, 
            self.recall_update_40, self.recall_update_50]

        return sess.run(recall_update_ops, feed_dict={ 
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[3],
            self.hist_i_new: batch[4],
            self.hist_t: batch[5],
            self.sl: batch[6],
            self.sl_new: batch[7], 
            self.is_training: False,       
        })


    def save(self, sess):
        checkpoint_path = os.path.join(self.config['model_dir'], 'TLSAN')
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path=checkpoint_path, global_step=self.global_step.eval())
        json.dump(self.config, open('%s-%d.json' % (checkpoint_path, self.global_step.eval()), 'w'), indent=2)
        print('model saved at %s' % save_path, flush=True)


    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path, flush=True)


def attention_net(
        user,
        enc, 
        enc_new, 
        sl,
        sl_new,
        rep_time, 
        gamma, 
        num_units, 
        num_heads, 
        num_blocks, 
        dropout_rate, 
        is_training, 
        reuse):
    with tf.variable_scope("all", reuse=reuse):
        with tf.variable_scope("long-term"):
            user_exp = tf.tile(tf.expand_dims(user, 1), [1, tf.shape(enc)[1], 1])
            enc = tf.multiply(user_exp, enc)
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks0_{}".format(i)):
                    with tf.variable_scope('fwbw_attention'):
                        fw_res = directional_attention_with_dense(
                            rep_tensor=enc, 
                            rep_time=rep_time, 
                            rep_length=sl,
                            gamma = gamma, 
                            direction='forward', 
                            scope='feature_wise_self_attention_fw',
                            keep_prob=1-dropout_rate, 
                            is_training=is_training, 
                            wd=0, 
                            activation='relu', 
                            tensor_dict=None, 
                            name='_fw_attention')
                        bw_res = directional_attention_with_dense(
                            rep_tensor=enc, 
                            rep_time=rep_time, 
                            rep_length=sl,
                            gamma = gamma, 
                            direction='backward', 
                            scope='feature_wise_self_attention_bw',
                            keep_prob=1-dropout_rate, 
                            is_training=is_training, 
                            wd=0, 
                            activation='relu', 
                            tensor_dict=None, 
                            name='_bw_attention')

                    with tf.variable_scope('long-term layer'):
                        enc, att0 = feature_wise_attention(
                            rep_tensor=tf.concat([fw_res, bw_res], -1), 
                            rep_length=sl, 
                            num_heads=num_heads, 
                            scope='feature_wise_attention1',
                            keep_prob=1-dropout_rate, 
                            is_training=is_training, 
                            wd=0, 
                            activation='relu', 
                            tensor_dict=None, 
                            name='long_attn')
        
                    enc = tf.expand_dims(tf.layers.dense(enc, num_units), 1)

        with tf.variable_scope("short-term"):
            enc = tf.concat([enc, enc_new], 1)
            user_exp = tf.tile(tf.expand_dims(user, 1), [1, tf.shape(enc)[1], 1])
            enc = tf.multiply(user_exp, enc)
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks1_{}".format(i)):
                    with tf.variable_scope("short-term layer"):
                        enc_new, att1 = feature_wise_attention(
                            rep_tensor=enc,
                            rep_length=sl_new+1, 
                            num_heads=num_heads, 
                            scope='feature_wise_attention2',
                            keep_prob=1-dropout_rate, 
                            is_training=is_training, 
                            wd=0, 
                            activation='relu', 
                            tensor_dict=None, 
                            name='short_attn')

        return enc_new, att0, att1

# --------------- supporting networks(modified) ----------------

def directional_attention_with_dense(rep_tensor, rep_time, rep_length, gamma, direction=None, scope=None,
                                     keep_prob=1., is_training=None, wd=0., activation='elu',
                                     tensor_dict=None, name=None):    
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    rep_mask = tf.sequence_mask(rep_length, sl)
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(tf.diag(0 - tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        #position
        length_col = tf.tile(tf.expand_dims(rep_time, -1), [1, 1, sl])
        length_row = tf.transpose(length_col, [0, 2, 1])
        position = - tf.abs(tf.subtract(length_col, length_row))
        position = tf.multiply(gamma, tf.cast(position, tf.float32))

        # non-linear
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                False, wd, keep_prob, is_training)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_training)

        # attention
        with tf.variable_scope('disan_attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec
            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,sl,sl,vec

            logits_masked = exp_mask_for_high_rank_position(logits, attn_mask, position)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('disan_output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_training) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_training) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)

        # save attention
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name + '_attn_score'] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


def feature_wise_attention(rep_tensor, rep_length, num_heads, scope=None,
                                keep_prob=1., is_training=None, wd=0., activation='elu',
                                tensor_dict=None, name=None):
    # Multi-heads
    rep_tensor = tf.concat(tf.split(rep_tensor, num_heads, axis=2), axis=0)  # bs*heads,sl,vec/heads
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    rep_mask = tf.sequence_mask(rep_length, sl)
    rep_mask = tf.tile(rep_mask, [num_heads, 1])
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'feature_wise_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_training)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_training)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs*heads,sl,vec/heads
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs*heads,vec/heads
        attn_output = tf.concat(tf.split(attn_output, num_heads, axis=0), axis=1)

        # save attention
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output, soft


def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_training=None):
    if is_training is None:
        is_training = False
    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_training)
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_training, scope='bn')
        return activation_func(linear_map)


def dropout(x, keep_prob, is_training, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        assert is_training is not None
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_training, lambda: d, lambda: x)
            return out
        return x


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, keep_prob=1.0,
           is_training=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]
    flat_args = [flatten(arg, 1) for arg in args] # for dense layer [(-1, d)]
    if keep_prob < 1.0:
        assert is_training is not None
        flat_args = [tf.cond(is_training, lambda: tf.nn.dropout(arg, keep_prob), lambda: arg)# for dense layer [(-1, d)]
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope) # dense
    out = reconstruct(flat_out, args[0], 1) # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out


def _linear(xs,output_size,bias,bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs,-1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size,output_size],dtype=tf.float32)
        if bias:
            bias = tf.get_variable('bias', shape=[output_size],dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')


def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER, 
                name=name or 'exp_mask_for_high_rank')


def exp_mask_for_high_rank_position(val, val_mask, position, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    position = tf.expand_dims(position, -1)
    return tf.add(tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER), position,
                name=name or 'exp_mask_for_high_rank_position')


def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg', var)
        counter += 1
    return counter
