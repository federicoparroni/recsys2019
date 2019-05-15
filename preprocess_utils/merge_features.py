import data
from tqdm import tqdm
import pandas as pd
import numpy as np

def find_last_clickout_indices(df):
    indices = []
    cur_ses = ''
    cur_user = ''
    temp_df = df[df.action_type == 'clickout item'][['user_id', 'session_id', 'action_type']]
    for idx in tqdm(temp_df.index.values[::-1]):
        ruid = temp_df.at[idx, 'user_id']
        rsid = temp_df.at[idx, 'session_id']
        if (ruid != cur_user or rsid != cur_ses):
            indices.append(idx)
            cur_user = ruid
            cur_ses = rsid
    return indices[::-1]


def expand_impressions(df):
    res_df = df.copy()
    res_df.impressions = res_df.impressions.str.split('|')
    res_df = res_df.reset_index()

    res_df = pd.DataFrame({
        col: np.repeat(res_df[col].values, res_df.impressions.str.len())
        for col in res_df.columns.drop('impressions')}
    ).assign(**{'impressions': np.concatenate(res_df.impressions.values)})[res_df.columns]

    res_df = res_df.rename(columns={'impressions': 'item_id'})
    res_df = res_df.astype({'item_id':'int'})
    return res_df[['index', 'step','user_id', 'session_id', 'item_id']]


"""
    given an array of features for ranking, it merges those in a single dataframe and returns
    a train and test df. the test df contains just the target sessions, identified by the target indices 
    in that mode and cluster, in the order in which the target indices are
"""
def merge_features(mode, cluster, features_array):

    # load the full_df
    train_df = data.train_df(mode, cluster)
    test_df = data.test_df(mode, cluster)
    full_df = pd.concat([train_df, test_df])
    del train_df, test_df

    # retrieve the indeces of the last clikcouts
    print('find_last_click_idxs')
    last_click_idxs=find_last_clickout_indices(full_df)

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
    train_df = expand_impressions(train_df)
    validation_test_df = expand_impressions(validation_test_df)

    # do the join
    print('join with the features')
    print(f'train_shape: {train_df.shape}\n vali_test_shape: {validation_test_df.shape}')
    for f in features_array:
        feature = f(mode=mode, cluster='no_cluster').read_feature(one_hot=True)
        print(f'len of feature:{len(feature)}')
        train_df = train_df.merge(feature)
        validation_test_df = validation_test_df.merge(feature)
        print(f'train_shape: {train_df.shape}\n vali_shape: {validation_test_df.shape}')

    print('sorting by index and step...')
    # sort the dataframes
    train_df.sort_values(['index', 'step'], inplace=True)
    validation_test_df.sort_values(['index', 'step'], inplace=True)

    print('after join')
    return train_df, validation_test_df