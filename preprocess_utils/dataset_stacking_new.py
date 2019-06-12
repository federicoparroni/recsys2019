from scipy.sparse import save_npz
import data
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from utils.check_folder import check_folder
from utils.menu import single_choice
from preprocess_utils.merge_features import merge_features
from os.path import join
from extract_features.impression_position_session import ImpressionPositionSession
from extract_features.label import ImpressionLabel
from extract_features.stacking.rnn_scores import ScoresRNN
from extract_features.stacking.xgboost_scores import ScoresXGB



def create_groups(df):
    df = df[['user_id', 'session_id']]
    group = df.groupby(['user_id', 'session_id'],
                       sort=False).apply(lambda x: len(x)).values
    return group

def create_dataset(mode, cluster):

    features_array = [ImpressionLabel, ImpressionPositionSession, ScoresRNN, ScoresXGB]

    train_df, test_df, train_idxs, _ = merge_features(mode, cluster, features_array, merge_kind='left')
    train_df = train_df.replace(-1, np.nan)
    test_df = test_df.replace(-1, np.nan)

    bp = 'dataset/preprocessed/{}/{}/stacking/'.format(cluster, mode)
    check_folder(bp)

    X_train = train_df.drop(
        ['index', 'user_id', 'session_id', 'item_id', 'label'], axis=1)
    X_train = X_train.to_sparse(fill_value=0)
    X_train = X_train.astype(np.float64)
    X_train = X_train.to_coo().tocsr()
    save_npz(join(bp, 'X_train'), X_train)
    print('X_train saved')

    y_train = train_df[['label']]
    y_train.to_csv(join(bp, 'y_train.csv'))
    print('y_train saved')

    group = create_groups(train_df)
    print(len(group))
    np.save(join(bp, 'group_train'), group)
    print('train groups saved')

    np.save(join(bp, 'train_indices'), train_idxs)

    print('train data completed')

    X_test = test_df.drop(
        ['index', 'user_id', 'session_id', 'item_id', 'label'], axis=1)

    # if mode == 'full':
    X_test = X_test.to_sparse(fill_value=0)
    X_test = X_test.astype(np.float64)
    X_test = X_test.to_coo().tocsr()
    save_npz(join(bp, 'X_test'), X_test)
    # else:
    #    X_test.to_csv(join(bp, 'X_test.csv'))
    print('X_test saved')

    y_test = test_df[['label']]
    y_test.to_csv(join(bp, 'y_test.csv'))
    print('y_test saved')

    group = create_groups(test_df)
    print(len(group))
    np.save(join(bp, 'group_test'), group)

    print('test groups saved')

    print('test data completed')

if __name__ == "__main__":

    from utils.menu import mode_selection
    from utils.menu import cluster_selection
    mode = mode_selection()
    cluster = cluster_selection()
    create_dataset(mode, cluster)
