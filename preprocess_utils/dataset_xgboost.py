from scipy.sparse import save_npz
import data
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle


def groups(users):
    au = users[0]
    groups = []
    count = 1
    for u in tqdm(users):
        if u != au:
            groups.append(count - 1)
            count = 1
            au = u
        count += 1
    groups.append(count - 1)
    return groups


def create_groups(df, mode, cluster):
    users = df['user_id'].to_dense().values
    print('data are ready')
    group = groups(list(users))
    np.save('dataset/preprocessed/{}/{}/xgboost/group'.format(cluster, mode), group)


def create_dataset(mode, cluster):
    # training
    train = data.classification_train_df(
        mode=mode, sparse=True, cluster=cluster, algo='xgboost')

    X_train = train.iloc[:, 3:]
    X_train = X_train.astype(np.float64)
    X_train = X_train.to_coo().tocsr()
    save_npz(
        'dataset/preprocessed/{}/{}/xgboost/X_train'.format(cluster, mode), X_train)
    print('X_train saved')

    y_train = train[['label']]
    y_train.to_csv(
        'dataset/preprocessed/{}/{}/xgboost/y_train.csv'.format(cluster, mode))
    print('y_train saved')

    create_groups(train, mode, cluster)
    print('groups saved')

    print('train data completed')

    # testing
    test = data.classification_test_df(
        mode=mode, sparse=True)
    test_scores = test[['user_id', 'session_id',
                        'impression_position']].to_dense()
    test_scores.to_csv(
        'dataset/preprocessed/{}/{}/xgboost/test_scores.csv'.format(cluster, mode))
    print('test_scores saved')

    d = {}
    for idx, row in test_scores.iterrows():
        sess_id = row['session_id']
        if sess_id in d:
            d[sess_id] += [idx]
        else:
            d[sess_id] = [idx]
    with open('dataset/preprocessed/{}/{}/xgboost/dict'.format(cluster, mode), 'wb') as f:
        pickle.dump(d, f)
    print('dict saved')

    X_test = test.iloc[:, 3:]
    X_test = X_test.to_coo().tocsr()
    save_npz('dataset/preprocessed/{}/{}/xgboost/X_test'.format(cluster, mode), X_test)
    print('X_test saved')

    print('test data completed')


if __name__ == "__main__":
    mode = 'small'
    cluster = 'no_cluster'
    create_dataset(mode, cluster)
