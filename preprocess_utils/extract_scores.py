import pandas as pd
import numpy as np
import data
import os
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices


def expand_item_recommendations(df, perScore=True):
    res_df = df.copy()

    if perScore == False:
        res_df.item_recommendations = res_df.item_recommendations.str.split(' ')

    res_df = res_df.reset_index()
    res_df = pd.DataFrame({
        col: np.repeat(res_df[col].values, res_df.item_recommendations.str.len())
        for col in res_df.columns.drop('item_recommendations')}
    ).assign(**{'item_recommendations': np.concatenate(res_df.item_recommendations.values)})[res_df.columns]

    res_df = res_df.rename(columns={'item_recommendations': 'item_id'})
    res_df = res_df.astype({'item_id': 'int'})
    return res_df[['index', 'item_id']]

def get_score(item, scores, rec):
    res = np.empty(item.shape)
    for i in range(len(item)):
        res[i] = scores[i][rec[i].index(item[i])]
    return res

def get_pos(item, rec):
    res = np.empty(item.shape)
    for i in range(len(item)):
        res[i] = rec[i].index(str(item[i])) + 1

    return res.astype(int)


def assign_score(df, name):
    print('Convert and adding submission scores positions..')
    df_t = expand_item_recommendations(df)

    df = pd.merge(df_t, df, on=['index'], how='left')
    df['score_' + name] = get_score(df['item_id'].values, df['scores'].values,
                                         df['item_recommendations'].values)
    df = df.drop(['scores', 'item_recommendations'], axis=1)
    return df

def train_indices(mode='local', cluster='no_cluster'):
    df_train = data.train_df(mode=mode, cluster=cluster)
    df_test = data.test_df(mode=mode, cluster=cluster)
    target_indices = data.target_indices(mode=mode, cluster=cluster)
    df = pd.concat([df_train, df_test])
    idx = find_last_clickout_indices(df)
    train_idx = set(idx) - set(target_indices)
    return train_idx