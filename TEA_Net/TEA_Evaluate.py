# @Author: Jinyu Zhang
# @Time: 2022/8/24 15:24
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

import math
import numpy as np


def evaluate_ratings(sess, GCN_net, uid, seq_A, seq_B, len_A, len_B, target_A, target_B, test_batch_num,
                     test_length):
    RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A = 0, 0, 0, 0, 0, 0
    RC_5_B, RC_10_B, RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B = 0, 0, 0, 0, 0, 0
    NDCG_5_A, NDCG_10_A, NDCG_20_A, NDCG_5_B, NDCG_10_B, NDCG_20_B = 0, 0, 0, 0, 0, 0

    for batch in range(test_batch_num):
        test_seq_A = seq_A[batch]
        test_seq_B = seq_B[batch]

        test_len_A = len_A[batch]
        test_len_B = len_B[batch]

        test_target_A = target_A[batch]
        test_target_B = target_B[batch]

        test_uid = uid[batch]

        prediction_A, prediction_B = \
            GCN_net.evaluate_gcn(sess, uid=test_uid, seq_A=test_seq_A, seq_B=test_seq_B,
                                 len_A=test_len_A, len_B=test_len_B,
                                 target_A=test_target_A, target_B=test_target_B, dropout_rate=0, keep_prob=1.0)
        # Calculate the score of Domain-A
        RC_A, MRR_A, NDCG_A = eval_score(prediction_A, test_target_A, [5, 10, 20])
        RC_5_A += RC_A[0]
        RC_10_A += RC_A[1]
        RC_20_A += RC_A[2]
        MRR_5_A += MRR_A[0]
        MRR_10_A += MRR_A[1]
        MRR_20_A += MRR_A[2]
        # new
        NDCG_5_A += NDCG_A[0]
        NDCG_10_A += NDCG_A[1]
        NDCG_20_A += NDCG_A[2]
        # Calculate the score of Domain-B
        RC_B, MRR_B, NDCG_B = eval_score(prediction_B, test_target_B, [5, 10, 20])
        RC_5_B += RC_B[0]
        RC_10_B += RC_B[1]
        RC_20_B += RC_B[2]
        MRR_5_B += MRR_B[0]
        MRR_10_B += MRR_B[1]
        MRR_20_B += MRR_B[2]
        # new
        NDCG_5_B += NDCG_B[0]
        NDCG_10_B += NDCG_B[1]
        NDCG_20_B += NDCG_B[2]

    return [RC_5_A / test_length, RC_10_A / test_length, RC_20_A / test_length,
            MRR_5_A / test_length, MRR_10_A / test_length, MRR_20_A / test_length,
            RC_5_B / test_length, RC_10_B / test_length, RC_20_B / test_length,
            MRR_5_B / test_length, MRR_10_B / test_length, MRR_20_B / test_length,
            NDCG_5_A / test_length, NDCG_10_A / test_length, NDCG_20_A / test_length,
            NDCG_5_B / test_length, NDCG_10_B / test_length, NDCG_20_B / test_length]


def eval_score(pred_list, target_list, options):
    RC, MRR, NDCG = [], [], []
    pred_list = pred_list.argsort()
    for k in options:  # k:5
        RC.append(0)
        MRR.append(0)
        NDCG.append(0)
        temp_list = pred_list[:, -k:]
        search_index = 0
        while search_index < len(target_list):
            pos = np.argwhere(temp_list[search_index] == target_list[search_index])
            if len(pos) > 0:
                RC[-1] += 1
                MRR[-1] += 1 / (k - pos[0][0])
                NDCG[-1] += math.log(2) / math.log(k - pos[0][0] + 2)
            else:
                RC[-1] += 0
                MRR[-1] += 0
                NDCG[-1] += 0
            search_index += 1
    return RC, MRR, NDCG
