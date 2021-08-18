import numpy as np
import pandas as pd
import bottleneck as bn
import scipy.stats

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Precision_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    precision = tmp / np.minimum(k, X_pred_binary.sum(axis=1))
    return precision

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

# 人均推荐列表中，物品的平均popularity
def ARP(predictions, sorted_movieId):
    arp = np.mean([np.mean(sorted_movieId.loc[row[1].to_numpy().tolist()[1]]) for row in predictions.iterrows()])
    return arp

# 被推荐的电影数量 占所有总电影数量的多少占比
def Agg_Div(predictions,sorted_movieId):
    movieIds_set = set()
    for row in predictions.iterrows():
        movieIds_set = movieIds_set.union(set(row[1]))
    return len(movieIds_set)/len(sorted_movieId)

def group_UDP(X,Y):
    assert len(X) == len(Y), "two distribution do not have same number of bins"
    n = len(X)
    sumShannon = 0.0
    for i in range(n):
        M_i = 0.5 * (X[i] + Y[i])
        sumShannon += 0.5 * scipy.stats.entropy(X[i],M_i) + 0.5 * scipy.stats.entropy(Y[i],M_i)
    return sumShannon/n