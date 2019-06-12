import data
from tqdm import tqdm
import pandas as pd
import numpy as np
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions

"""
    given an array of features for ranking, it merges those in a single dataframe and returns
    a train and test df. the test df contains just the target sessions, identified by the target indices 
    in that mode and cluster, in the order in which the target indices are
"""
def merge_features(mode, cluster, features_array, onehot=True, merge_kind='inner'):
    n_nans_train = 0
    n_nans_validation = 0

    # load the full_df
    train_df = data.train_df(mode, cluster)
    test_df = data.test_df(mode, cluster)
    full_df = pd.concat([train_df, test_df])
    del train_df, test_df

    # retrieve the indeces of the last clikcouts
    print('find_last_click_idxs')
    last_click_idxs=find_last_clickout_indices(full_df)
    last_click_idxs = sorted(last_click_idxs)

    # filter on the found indeces obtaining only the rows of a last clickout
    print('filter full on last click idxs')
    click_df = full_df.loc[last_click_idxs].copy()

    print('retrieve vali_idxs')
    # if the mode is full we don't have the validation if the mode is small or local the validation is performed
    # on the target indices

    vali_test_idxs = data.target_indices(mode, cluster)


    # construct the validation train and test df_base
    print('construct test and vali df')
    validation_test_df = click_df.loc[vali_test_idxs]

    all_idxs = click_df.index.values

    # find the differences
    print('construct train df')
    train_idxs = np.setdiff1d(all_idxs, vali_test_idxs, assume_unique=True)
    train_df = click_df.loc[train_idxs]

    # expand the impression as rows
    print('expand the impression')
    train_df = expand_impressions(train_df)[['user_id', 'session_id', 'item_id', 'index']]
    train_df['dummy_step']=np.arange(len(train_df))
    validation_test_df = expand_impressions(validation_test_df)[['user_id', 'session_id', 'item_id', 'index']]
    validation_test_df['dummy_step'] = np.arange(len(validation_test_df))

    # do the join
    print('join with the features')
    print(f'train_shape: {train_df.shape}\n vali_test_shape: {validation_test_df.shape}')
    for f in features_array:
        if type(f) == tuple:
            feature = f[0](mode=mode, cluster='no_cluster').read_feature(one_hot=f[1])
        else:
            feature = f(mode=mode, cluster='no_cluster').read_feature(one_hot=onehot)
        print(f'len of feature:{len(feature)}')
        train_df = train_df.merge(feature, how=merge_kind)
        validation_test_df = validation_test_df.merge(feature, how=merge_kind)
        print(f'train_shape: {train_df.shape}\n vali_shape: {validation_test_df.shape}')
        
        if merge_kind == 'left':
            delta_nan_train = train_df[train_df.columns[-len(feature.columns):]].isnull().sum().sum()
            print('train: num columns of feature: {}. nans introduced: {}'.format(len(feature.columns), delta_nan_train))
            delta_nan_validation = validation_test_df[validation_test_df.columns[-len(feature.columns):]].isnull().sum().sum() - n_nans_validation
            print('validation: num columns of feature: {}. nans introduced: {}'.format(len(feature.columns), delta_nan_validation))
            print('\n')

    print('sorting by index and step...')
    # sort the dataframes
    train_df.sort_values(['index', 'dummy_step'], inplace=True)
    train_df.drop('dummy_step', axis=1, inplace=True
                  )
    validation_test_df.sort_values(['index', 'dummy_step'], inplace=True)
    validation_test_df.drop('dummy_step', axis=1, inplace=True)

    print('after join')
    return train_df, validation_test_df, train_idxs, vali_test_idxs
