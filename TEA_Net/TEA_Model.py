# @Author: Jinyu Zhang
# @Time: 2022/8/24 9:31
# @E-mail: JinyuZ1996@outlook.com

import os
import random
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from TEA_Net.TEA_Setting import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 2023
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
args = Settings()


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def optimizer(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                        grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
    return train_op


class TEA_Net:
    def __init__(self, n_items_A, n_items_B, n_users, graph_main, graph_A, graph_B):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.n_items_A = n_items_A
        self.n_items_B = n_items_B
        self.n_users = n_users
        self.graph_main = graph_main
        self.graph_A = graph_A
        self.graph_B = graph_B
        self.embedding_size = args.embedding_size
        self.n_fold = args.n_fold
        self.alpha = args.alpha
        self.temperature = args.beta
        self.layer_size = args.layer_size
        self.regular_rate_att = args.regular_rate_att
        self.num_heads = args.num_heads
        self.n_layers = args.num_layers
        self.lr_A = args.lr_A
        self.lr_B = args.lr_B
        self.l2_regular_rate = args.l2_regular_rate
        self.dim_coefficient = args.dim_coefficient
        self.batch_size = args.batch_size
        self.weight_size = eval(self.layer_size)
        self.is_training = True
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.uid, self.seq_A, self.seq_B, self.len_A, self.len_B, self.target_A, \
                self.target_B, self.dropout_rate, self.keep_prob = self.get_inputs()

            with tf.name_scope('encoder'):
                self.all_weights = self._init_weights()
                self.i_embeddings_A, self.u_embeddings, self.i_embeddings_B, self.ssl_loss_A, self.ssl_loss_B = \
                    self.graph_encoder(self.n_items_A, self.n_users, self.n_items_B, self.graph_main, self.graph_A,
                                       self.graph_B)
                self.seq_emb_A_output, self.seq_emb_B_output = self.seq_encoder(self.uid, self.seq_A, self.seq_B,
                                                                                self.dropout_rate, self.i_embeddings_A,
                                                                                self.u_embeddings, self.i_embeddings_B)
            with tf.name_scope('prediction_A'):
                self.pred_A = self.prediction_A(self.n_items_A, self.seq_emb_B_output, self.seq_emb_A_output,
                                                self.keep_prob)
            with tf.name_scope('prediction_B'):
                self.pred_B = self.prediction_B(self.n_items_B, self.seq_emb_A_output, self.seq_emb_B_output,
                                                self.keep_prob)
            with tf.name_scope('loss'):
                self.loss_A, self.loss_B, self.loss = self.cal_loss(self.target_A, self.pred_A, self.target_B,
                                                                    self.pred_B)
            with tf.name_scope('optimizer'):
                self.train_op_A = optimizer(self.loss_A + self.ssl_loss_A, self.lr_A)
                self.train_op_B = optimizer(self.loss_B + self.ssl_loss_B, self.lr_B)

            total_params = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()  # 获取变量的形状
                params = 1
                for dim in shape:
                    params *= dim.value  # 计算形状的元素数目
                total_params += params

            print("Trainable param:{}".format(total_params))

    def get_inputs(self):
        uid = tf.placeholder(dtype=tf.int32, shape=[None, ], name='uid')
        seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_A')
        seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_B')
        len_A = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='len_A')
        len_B = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='len_B')
        target_A = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_A')
        target_B = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_B')
        dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        return uid, seq_A, seq_B, len_A, len_B, target_A, target_B, dropout_rate, keep_prob

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.embedding_size]))
        all_weights['item_embedding_A'] = tf.Variable(initializer([self.n_items_A, self.embedding_size]))
        all_weights['item_embedding_B'] = tf.Variable(initializer([self.n_items_B, self.embedding_size]))
        all_weights['shared_mk_A'] = tf.Variable(initializer(
            [self.num_heads, self.embedding_size * self.dim_coefficient // self.num_heads, self.embedding_size]))
        all_weights['shared_mv_A'] = tf.Variable(initializer(
            [self.num_heads, self.embedding_size, self.embedding_size]))
        all_weights['shared_mk_B'] = tf.Variable(initializer(
            [self.num_heads, self.embedding_size * self.dim_coefficient // self.num_heads, self.embedding_size]))
        all_weights['shared_mv_B'] = tf.Variable(initializer(
            [self.num_heads, self.embedding_size, self.embedding_size]))
        return all_weights

    def unzip_laplace(self, X):
        unzip_info = []
        fold_len = (X.shape[0]) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = X.shape[0]
            else:
                end = (i_fold + 1) * fold_len

            unzip_info.append(_convert_sp_mat_to_sp_tensor(X[start:end]))
        return unzip_info

    def graph_encoder(self, n_items_A, n_users, n_items_B, graph_main, graph_A, graph_B):

        main_info = self.unzip_laplace(graph_main)
        A_info = self.unzip_laplace(graph_A)
        B_info = self.unzip_laplace(graph_B)

        ego_ebd_main = tf.concat([self.all_weights['item_embedding_A'], self.all_weights['user_embedding'],
                                  self.all_weights['item_embedding_B']], axis=0)
        ego_ebd_A = tf.concat([self.all_weights['item_embedding_A'], self.all_weights['user_embedding']], axis=0)
        ego_ebd_B = tf.concat([self.all_weights['item_embedding_B'], self.all_weights['user_embedding']], axis=0)

        all_ebd_main = [ego_ebd_main]
        all_ebd_A = [ego_ebd_A]
        all_ebd_B = [ego_ebd_B]

        # MLAP for both domain A and B
        for k in range(0, self.n_layers):
            temp_A, temp_B = [], []
            for f in range(args.n_fold):
                temp_A.append(tf.sparse_tensor_dense_matmul(A_info[f], ego_ebd_A))
                temp_B.append(tf.sparse_tensor_dense_matmul(B_info[f], ego_ebd_B))

            # sum messages of neighbors.
            side_ebd_A = tf.concat(temp_A, 0)
            side_ebd_B = tf.concat(temp_B, 0)

            all_ebd_A += [side_ebd_A]
            all_ebd_B += [side_ebd_B]

        all_ebd_A = tf.stack(all_ebd_A, 1)
        all_ebd_A = tf.reduce_mean(all_ebd_A, axis=1, keepdims=False)
        node_A_A, node_A_U = tf.split(all_ebd_A, [n_items_A, n_users], 0)

        all_ebd_B = tf.stack(all_ebd_B, 1)
        all_ebd_B = tf.reduce_mean(all_ebd_B, axis=1, keepdims=False)
        node_B_B, node_B_U = tf.split(all_ebd_B, [n_items_B, n_users], 0)

        # SLAP for user-item cross-domain graph
        temp_main = []
        for f in range(args.n_fold):
            temp_main.append(tf.sparse_tensor_dense_matmul(main_info[f], ego_ebd_main))

        side_ebd_main = tf.concat(temp_main, 0)

        all_ebd_main += [side_ebd_main]

        all_ebd_main = tf.stack(all_ebd_main, 1)
        all_ebd_main = tf.reduce_mean(all_ebd_main, axis=1, keepdims=False)
        node_main_A, node_main_U, node_main_B = tf.split(all_ebd_main, [n_items_A, n_users, n_items_B], 0)

        # domain A shared external attention
        EA_ebd_A_A = self.node_external_attention(node_A_A, self.all_weights['shared_mk_A'],
                                                  self.all_weights['shared_mv_A'], self.dim_coefficient)
        # add & norm
        refined_A_A = tf.add(EA_ebd_A_A, node_A_A)
        normed_A_A = tf.contrib.layers.layer_norm(refined_A_A)

        EA_main_A = self.node_external_attention(node_main_A, self.all_weights['shared_mk_A'],
                                                 self.all_weights['shared_mv_A'], self.dim_coefficient)
        # add & norm
        refined_main_A = tf.add(EA_main_A, node_main_A)
        normed_main_A = tf.contrib.layers.layer_norm(refined_main_A)

        sum_ebd_A = normed_A_A + normed_main_A

        # domain B shared external attention
        EA_ebd_B_B = self.node_external_attention(node_B_B, self.all_weights['shared_mk_B'],
                                                  self.all_weights['shared_mv_B'], self.dim_coefficient)
        # add & norm
        refined_B_B = tf.add(EA_ebd_B_B, node_B_B)
        normed_B_B = tf.contrib.layers.layer_norm(refined_B_B)

        EA_ebd_main_B = self.node_external_attention(node_main_B, self.all_weights['shared_mk_B'],
                                                  self.all_weights['shared_mv_B'], self.dim_coefficient)
        # add & norm
        refined_main_B = tf.add(EA_ebd_main_B, node_main_B)
        normed_main_B = tf.contrib.layers.layer_norm(refined_main_B)

        sum_ebd_B = normed_B_B + normed_main_B

        sum_ebd_U = node_A_U + node_B_U + node_main_U

        ssl_loss_A = self.cal_loss_ssl(node_A_U, node_main_U) + self.cal_loss_ssl(node_A_A, node_main_A)
        ssl_loss_B = self.cal_loss_ssl(node_B_U, node_main_U) + self.cal_loss_ssl(node_B_B, node_main_B)

        return sum_ebd_A, sum_ebd_U, sum_ebd_B, ssl_loss_A, ssl_loss_B

    def cal_loss_ssl(self, ebd_in_main, ebd_in_sub):
        ebd_in_maingraph = tf.nn.l2_normalize(ebd_in_main, 1)
        ebd_in_subgraph = tf.nn.l2_normalize(ebd_in_sub, 1)

        similarity = tf.matmul(ebd_in_maingraph, ebd_in_subgraph, transpose_b=True)

        positive_logits = tf.linalg.diag_part(similarity) / self.temperature
        batch_size = tf.shape(ebd_in_maingraph)[0]
        diagonal_zeros = tf.zeros(batch_size, dtype=tf.float32)

        negative_logits = tf.linalg.set_diag(similarity, diagonal_zeros) / self.temperature

        num_negative = batch_size - 1
        positive_loss = -tf.math.log(tf.nn.sigmoid(positive_logits))
        negative_loss = -tf.reduce_sum(tf.math.log(1 - tf.nn.sigmoid(negative_logits)), axis=1)

        num_positive = tf.cast(1, tf.float32)
        num_negative = tf.cast(num_negative, tf.float32)

        # Compute the final loss
        info_nce_loss = (positive_loss + negative_loss) / (num_positive + num_negative)
        info_nce_loss = tf.reduce_mean(info_nce_loss)

        return info_nce_loss

    def seq_encoder(self, uid, seq_A, seq_B, dropout_rate, i_embeddings_A, u_embeddings, i_embeddings_B):
        with tf.variable_scope('seq_encoder'):
            self.user_embed = tf.nn.embedding_lookup(u_embeddings, uid)
            # domain A:
            item_embed_A = tf.nn.embedding_lookup(i_embeddings_A, seq_A)
            self.seq_embed_A = tf.reduce_max((item_embed_A), 1)
            seq_emb_A_output = tf.concat([self.seq_embed_A, self.user_embed], axis=1)
            seq_emb_A_output = tf.layers.dropout(seq_emb_A_output, rate=dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))
            # domain B
            item_embed_B = tf.nn.embedding_lookup(i_embeddings_B, seq_B)
            self.seq_embed_B = tf.reduce_max((item_embed_B), 1)
            seq_emb_B_output = tf.concat([self.seq_embed_B, self.user_embed], axis=1)
            seq_emb_B_output = tf.layers.dropout(seq_emb_B_output, rate=dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))

        return seq_emb_A_output, seq_emb_B_output

    def prediction_A(self, n_items_A, seq_emb_B_output, seq_emb_A_output, keep_prob):
        with tf.variable_scope('prediction_A'):
            concat_output = tf.concat([seq_emb_B_output, seq_emb_A_output], axis=-1)
            concat_output = tf.nn.dropout(concat_output, keep_prob)
            pred_A = tf.layers.dense(concat_output, n_items_A, activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(
                                         uniform=False))

            return pred_A

    def prediction_B(self, n_items_B, seq_emb_A_output, seq_emb_B_output, keep_prob):

        with tf.variable_scope('prediction_B'):
            concat_output = tf.concat([seq_emb_A_output, seq_emb_B_output], axis=-1)
            concat_output = tf.nn.dropout(concat_output, keep_prob)
            pred_B = tf.layers.dense(concat_output, n_items_B, activation=tf.nn.leaky_relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(
                                         uniform=False))
            return pred_B

    def cal_loss(self, target_A, pred_A, target_B, pred_B):

        loss_A1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_A, logits=pred_A)
        loss_A1 = tf.reduce_mean(loss_A1, name='loss_A')
        loss_A2 = self.l2_regular_rate * tf.reduce_sum(tf.square(self.seq_embed_A)) + \
                  self.l2_regular_rate * tf.reduce_sum(tf.square(self.user_embed))
        loss_A = loss_A1 + loss_A2

        loss_B1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_B, logits=pred_B)
        loss_B1 = tf.reduce_mean(loss_B1, name='loss_B')
        loss_B2 = self.l2_regular_rate * tf.reduce_sum(tf.square(self.seq_embed_B)) + \
                  self.l2_regular_rate * tf.reduce_sum(tf.square(self.user_embed))
        loss_B = loss_B1 + loss_B2

        loss_sum = loss_A + loss_B

        return loss_A, loss_B, loss_sum

    def node_external_attention(self, graph_node_embeddings, matrix_key, matrix_value, dimension_coefficient):
        element_size = tf.shape(graph_node_embeddings)[0]
        initial_query = tf.keras.layers.Dense(self.embedding_size * dimension_coefficient, activation=None,
                                              kernel_initializer='glorot_uniform')(graph_node_embeddings)
        multi_head_query = tf.reshape(initial_query, (self.num_heads, element_size,
                                                      self.embedding_size * dimension_coefficient // self.num_heads))
        attention_maps = tf.matmul(multi_head_query, matrix_key)  # A = Q · M_k
        normed_ebd = tf.nn.softmax(attention_maps, axis=2)  # ∑ each row of the M_k for normalization
        A_Mv = tf.matmul(normed_ebd, matrix_value)  # score = A · M_v
        merge_heads = tf.reduce_sum(A_Mv, axis=0)
        ebd_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)(merge_heads)

        return ebd_dropout

    def train_GCN(self, sess, uid, seq_A, seq_B, len_A, len_B, target_A, target_B,
                  dropout_rate, keep_prob):

        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B, self.len_A: len_A, self.len_B: len_B,
                     self.target_A: target_A, self.target_B: target_B,
                     self.dropout_rate: dropout_rate, self.keep_prob: keep_prob}

        return sess.run([self.loss_A, self.loss_B, self.train_op_A, self.train_op_B], feed_dict)

    def evaluate_gcn(self, sess, uid, seq_A, seq_B, len_A, len_B, target_A, target_B,
                     dropout_rate, keep_prob):
        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B,
                     self.len_A: len_A, self.len_B: len_B,
                     self.target_A: target_A, self.target_B: target_B, self.dropout_rate: dropout_rate,
                     self.keep_prob: keep_prob}
        return sess.run([self.pred_A, self.pred_B], feed_dict)
