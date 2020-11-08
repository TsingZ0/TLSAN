import os
import json
import numpy as np
import tensorflow as tf
from functools import reduce
from operator import mul

VERY_BIG_NUMBER = 1e30
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

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

        # [B] item label
        self.y = tf.placeholder(tf.float32, [None,])

        # [B, T] user's history item id
        self.hist_i = tf.placeholder(tf.int32, [None, None])

        # [B, T] user's history item purchase time
        self.hist_t = tf.placeholder(tf.int32, [None, None])

        # [B] valid length of `hist_i`
        self.sl = tf.placeholder(tf.int32, [None,])

        # learning rate
        self.lr = tf.placeholder(tf.float64, [])

        # whether it's training or not
        self.is_training = tf.placeholder(tf.bool, [])


    def build_model(self):
        item_emb = tf.get_variable(
            "item_emb", 
            [self.config['item_count'], self.config['itemid_embedding_size']])
        item_b = tf.get_variable(
            "item_b",
            [self.config['item_count'],],
            initializer=tf.constant_initializer(0.0))
        i_b = tf.gather(item_b, self.i)


        i_emb = tf.nn.embedding_lookup(item_emb, self.i)
        h_emb = tf.nn.embedding_lookup(item_emb, self.hist_i)

        num_blocks = self.config['num_blocks']
        num_heads = self.config['num_heads']
        dropout_rate = self.config['dropout']
        num_units = h_emb.get_shape().as_list()[-1]

        u_emb, self.att_vec, self.stt_vec = attention_net(
            h_emb,
            self.sl,
            self.hist_t,
            i_emb,
            num_units,
            num_heads,
            num_blocks,
            dropout_rate,
            self.is_training,
            False)

        self.logits = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1)

        # Eval
        self.eval_logits = tf.matmul(u_emb, item_emb, transpose_b=True) + item_b
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

        # loss
        l2_norm = tf.add_n([
            tf.nn.l2_loss(item_emb),
        ])
        
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
            ) + self.config['regulation_rate'] * l2_norm

        self.train_summary = tf.summary.merge([
            tf.summary.histogram('embedding/1_item_emb', item_emb),
            tf.summary.histogram('embedding/2_history_emb', h_emb),
            tf.summary.histogram('attention_output', u_emb),
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
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.config['max_gradient_norm'])

        # Update the model
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)
        
    
    def train(self, sess, batch, lr, add_summary=False):

        input_feed = {
            self.u: batch[0],
            self.i: batch[1],
            self.y: batch[2],
            self.hist_i: batch[3],
            self.hist_t: batch[4],
            self.sl: batch[5],
            self.lr: lr,
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


    def eval_auc(self, sess, batch):
        #positive_item_list
        res1 = sess.run(self.logits, feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[3],
            self.hist_t: batch[4],
            self.sl: batch[5],
            self.is_training: False,
        })
        #negative_item_list
        res2 = sess.run(self.logits, feed_dict={
            self.u: batch[0],
            self.i: batch[2],
            self.hist_i: batch[3],
            self.hist_t: batch[4],
            self.sl: batch[5],
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
            self.hist_t: batch[4],
            self.sl: batch[5],
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
            self.hist_t: batch[4],
            self.sl: batch[5],
            self.is_training: False,
        })


    def save(self, sess):
        checkpoint_path = os.path.join(self.config['model_dir'], 'csan')
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



def attention_net(
        enc, 
        sl,
        rep_time, 
        dec, 
        num_units, 
        num_heads, 
        num_blocks, 
        dropout_rate, 
        is_training, 
        reuse):
    with tf.variable_scope("all", reuse=reuse):
        with tf.variable_scope("feature_wise_self_attention"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks0_{}".format(i)):
                    with tf.variable_scope('fwbw_attention'):
                        fw_res = directional_attention_with_dense(
                            rep_tensor=enc, 
                            rep_time=rep_time, 
                            rep_length=sl,
                            direction='forward', 
                            scope='dir_attn_fw',
                            keep_prob=1 - dropout_rate, 
                            is_train=is_training, 
                            wd=0, 
                            activation='relu', 
                            tensor_dict=None, 
                            name='_fw_attn')
                        bw_res = directional_attention_with_dense(
                            rep_tensor=enc, 
                            rep_time=rep_time, 
                            rep_length=sl,
                            direction='backward', 
                            scope='dir_attn_bw',
                            keep_prob=1 - dropout_rate, 
                            is_train=is_training, 
                            wd=0, 
                            activation='relu', 
                            tensor_dict=None, 
                            name='_bw_attn')

                    with tf.variable_scope('feature_wise_self_attention'):
                        enc, att_vec = feature_wise_self_attention(
                            rep_tensor=tf.concat([fw_res, bw_res], -1), 
                            rep_length=sl, 
                            scope='feature_wise_self_attention',
                            keep_prob=1 - dropout_rate, 
                            is_train=is_training, 
                            wd=0, 
                            activation='relu', 
                            tensor_dict=None, 
                            name='_attn')
                    
                    enc = tf.layers.dense(enc, num_units)

        with tf.variable_scope("vanilla-attention"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks0_{}".format(i)):
                    dec, stt_vec = vanilla_attention(
                        queries=dec, 
                        keys=enc, 
                        keys_length=sl)

        return dec, att_vec, stt_vec

def vanilla_attention(queries, keys, keys_length):
    '''
    Inputs
        queries:     [B, H]
        keys:        [B, T, H]
        keys_length: [B]

    Returns
        output       [B, H]
    '''
    queries = tf.expand_dims(queries, 1) # [B, 1, H]
    # Multiplication
    outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) # [B, 1, T]

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
    key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]
    stt_vec = outputs

    # Weighted sum
    outputs = tf.squeeze(tf.matmul(outputs, keys))  # [B, H]

    return outputs, stt_vec


# --------------- supporting networks ----------------

def directional_attention_with_dense(rep_tensor, rep_time, rep_length, direction=None, scope=None,
                                     keep_prob=1., is_train=None, wd=0., activation='elu',
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


        # non-linear
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

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
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name + '_attn_score'] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


def feature_wise_self_attention(rep_tensor, rep_length, scope=None,
                                keep_prob=1., is_train=None, wd=0., activation='elu',
                                tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    rep_mask = tf.sequence_mask(rep_length, sl)
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'feature_wise_self_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = soft * rep_tensor  # bs, sl, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output, soft


def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
        is_train = False
    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    elif activation == 'selu':
        activation_func = selu
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_train, scope='bn')
        return activation_func(linear_map)


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        assert is_train is not None
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]
    flat_args = [flatten(arg, 1) for arg in args] # for dense layer [(-1, d)]
    if keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, keep_prob), lambda: arg)# for dense layer [(-1, d)]
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
        tf.add_to_collection(
                "weights_l2",
                tf.nn.l2_loss(W))
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
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
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
    return tf.add(tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER), tf.cast(position, tf.float32),
                name=name or 'exp_mask_for_high_rank_position')


def selu(x):
    with tf.name_scope('elu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg', var)
        counter += 1
    return counter
