import sys as sys
import numpy as np
import pandas as pd
from scipy import sparse


def split_train_test_proportion(data, test_prop=0.2, randomSeed=98765, verbose=False):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(randomSeed)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if verbose:
            if i % 1000 == 0:
                print("%d users sampled" % i)
                sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def load_train_data(csv_file,n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['user_index'].max() + 1

    rows, cols = tp['user_index'], tp['movie_index']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data

def load_tr_te_data(csv_file_tr, csv_file_te,n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['user_index'].min(), tp_te['user_index'].min())
    end_idx = max(tp_tr['user_index'].max(), tp_te['user_index'].max())

    rows_tr, cols_tr = tp_tr['user_index'] - start_idx, tp_tr['movie_index']
    rows_te, cols_te = tp_te['user_index'] - start_idx, tp_te['movie_index']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te