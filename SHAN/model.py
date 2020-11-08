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

        # [B, T] history item id
        self.hist_i = tf.placeholder(tf.int32, [None, None])
        self.hist_i_new = tf.placeholder(tf.int32, [None, None])

        # [B] valid length of `hist_i`
        self.sl = tf.placeholder(tf.int32, [None,])
        self.sl_new = tf.placeholder(tf.int32, [None,])

        # learning rate
        self.lr = tf.placeholder(tf.float32, [])

        # whether it's training or not
        self.is_training = tf.placeholder(tf.bool, [])


    def build_model(self):
        # item ID embedding
        item_emb = tf.get_variable(
            "item_emb", 
            [self.config['item_count'], self.config['embedding_size']])
        item_b = tf.get_variable(
            "item_b",
            [self.config['item_count'],],
            initializer=tf.constant_initializer(0.0))
        # user ID embedding
        user_emb = tf.get_variable(
            "user_emb", 
            [self.config['user_count'], self.config['embedding_size']])
        # parameter
        layer1_w = tf.get_variable(
            'layer1_w', 
            [self.config['embedding_size'], self.config['embedding_size']])
        layer2_w = tf.get_variable(
            'layer2_w', 
            [self.config['embedding_size'], self.config['embedding_size']])
        layer1_b = tf.get_variable(
            'layer1_b', 
            [1, self.config['embedding_size']])
        layer2_b = tf.get_variable(
            'layer2_b', 
            [1, self.config['embedding_size']])


        # item embedding
        i_emb = tf.nn.embedding_lookup(item_emb, self.i)
        i_b = tf.gather(item_b, self.i)

        # user embedding
        u_emb = tf.nn.embedding_lookup(user_emb, self.u)

        # history embedding
        h_emb = tf.nn.embedding_lookup(item_emb, self.hist_i)
        h_emb_new = tf.nn.embedding_lookup(item_emb, self.hist_i_new)

        dropout_rate = self.config['dropout']


        u_hybrid = attention_net(
            user_embedding=u_emb, 
            pre_sessions_embedding=h_emb, 
            current_session_embedding=h_emb_new, 
            layer1_w=layer1_w, 
            layer2_w=layer2_w, 
            layer1_b=layer1_b, 
            layer2_b=layer2_b, 
            reuse=False)

        self.logits = tf.reduce_sum(tf.multiply(u_hybrid, i_emb), -1) + i_b

        # Eval
        self.eval_logits = tf.matmul(u_hybrid, item_emb, transpose_b=True) + item_b
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
            tf.nn.l2_loss(layer1_w),
            tf.nn.l2_loss(layer2_w),
        ])

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))\
             + self.config['regulation_rate'] * l2_norm

        self.train_summary = tf.summary.merge([
            tf.summary.histogram('embedding/1_item_emb', item_emb),
            tf.summary.histogram('embedding/2_user_emb', user_emb),
            tf.summary.histogram('embedding/3_history_emb', h_emb),
            tf.summary.histogram('embedding/4_history_emb_new', h_emb_new),
            tf.summary.histogram('attention_output', u_hybrid),
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
            self.sl: batch[5],
            self.sl_new: batch[6],
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
            self.sl: batch[5],
            self.sl_new: batch[6], 
            self.is_training: False,
        })
        #negative_item_list
        res2 = sess.run(self.logits, feed_dict={
            self.u: batch[0],
            self.i: batch[2],
            self.hist_i: batch[3],
            self.hist_i_new: batch[4],
            self.sl: batch[5],
            self.sl_new: batch[6],
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
            self.sl: batch[5],
            self.sl_new: batch[6], 
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
            self.sl: batch[5],
            self.sl_new: batch[6], 
            self.is_training: False,       
        })


    def save(self, sess):
        checkpoint_path = os.path.join(self.config['model_dir'], 'shan')
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path=checkpoint_path, global_step=self.global_step.eval())
        json.dump(self.config, open('%s-%d.json' % (checkpoint_path, self.global_step.eval()), 'w'), indent=2)
        print('model saved at %s' % save_path, flush=True)


    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path, flush=True)


def attention_net(
        user_embedding, 
        pre_sessions_embedding, 
        current_session_embedding, 
        layer1_w, 
        layer2_w, 
        layer1_b, 
        layer2_b, 
        reuse):
    with tf.variable_scope("all", reuse=reuse):
        dim0 = tf.shape(user_embedding)[0]
        layer1_w = tf.tile(tf.expand_dims(layer1_w, 0), [dim0, 1, 1])
        layer2_w = tf.tile(tf.expand_dims(layer2_w, 0), [dim0, 1, 1])
        layer1_b = tf.tile(tf.expand_dims(layer1_b, 0), [dim0, 1, 1])
        layer2_b = tf.tile(tf.expand_dims(layer2_b, 0), [dim0, 1, 1])

        user_embedding = tf.expand_dims(user_embedding, 1)
        with tf.variable_scope("long-term"):
            long_user_embedding = attention_layer1(
                user_embedding, 
                pre_sessions_embedding,
                layer1_w, 
                layer1_b)

        long_user_embedding = tf.expand_dims(long_user_embedding, 1)
        with tf.variable_scope("short-term"):
            hybrid_user_embedding = attention_layer2(
                long_user_embedding, 
                current_session_embedding,
                layer2_w, 
                layer2_b)

        return hybrid_user_embedding

# -----------------reference(modified)------------------

def attention_layer1(user_embedding, pre_sessions_embedding, layer1_w, layer1_b):
    weight = tf.nn.softmax(tf.matmul(user_embedding, 
        tf.transpose(tf.sigmoid(tf.matmul(pre_sessions_embedding, layer1_w) + layer1_b), [0, 2, 1])))

    out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.transpose(weight, [0, 2, 1])), axis=1)
    return out

def attention_layer2(long_user_embedding, current_session_embedding, layer2_w, layer2_b):
    session_embedding = tf.concat([current_session_embedding, long_user_embedding], 1)

    weight = tf.nn.softmax(tf.matmul(long_user_embedding,
        tf.transpose(tf.sigmoid(tf.matmul(session_embedding, layer2_w) + layer2_b), [0, 2, 1])))

    out = tf.reduce_sum(tf.multiply(session_embedding, tf.transpose(weight, [0, 2, 1])), axis=1)
    return out

