import pandas as pd
import numpy as np
import data
import os
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from tqdm import tqdm

full_df = data.full_df()

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

def convert_and_assign_score(df, name):
    print('Convert and adding submission scores positions..')
    df_t = expand_item_recommendations(df)

    df = pd.merge(df_t, df, on=['index'], how='left')
    df['score_' + name] = get_score(df['item_id'].values, df['scores'].values,
                                         df['item_recommendations'].values)
    df = df.drop(['scores', 'item_recommendations'], axis=1)
    return df

def convert_and_add_labels(df, target_indices, name='', mode='local'):
    # full = data.full_df()
    full = full_df
    # target_indices = data.target_indices(mode=mode, cluster='no_cluster')

    full = full.loc[target_indices]
    full['index'] = list(full.index)
    full['impressions'] = full['impressions'].str.split('|')
    full = full[['reference', 'index', 'impressions']]

    # df = pd.merge(df, full, on=['user_id', 'session_id'])
    df = pd.merge(df, full, on=['index'], how='left')

    df = convert_and_assign_score(df, name)

    print('Adding labels..')
    print(df.columns)

    # df['label'] = df.progress_apply(lambda x: 1 if str(x['reference']) == str(x['item_id']) else 0, axis=1)

    df['label'] = np.where(df['reference'].astype(str) == df['item_id'].astype(str), 1, 0)

    df['impression_position'] = get_pos(df['item_id'].values, df['impressions'].values)

    df = df.drop(['reference', 'impressions'], axis=1)
    return df


def train_indices(mode='local', cluster='no_cluster'):
    df_train = data.train_df(mode=mode, cluster=cluster)
    df_test = data.test_df(mode=mode, cluster=cluster)
    target_indices = data.target_indices(mode=mode, cluster=cluster)
    df = pd.concat([df_train, df_test])
    idx = find_last_clickout_indices(df)
    train_idx = set(idx) - set(target_indices)
    return train_idx


def create_dataset(mode='local', cluster='no_cluster', name='stack_train', directory='', isTrain=True, save_path=''):
    print('Dataset creation')
    #check directory
    if not os.path.isdir(directory):
        print("Error: no directory founded")
        return

    if isTrain:
        t_idx = list(train_indices(mode=mode, cluster=cluster))
    else:
        t_idx = list(data.target_indices(mode=mode, cluster=cluster))
    dataframes = []
    dataframes_name = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if str(file) == '.DS_Store':
                continue

            if file.endswith('.npy'):
                f = np.load(directory + '/' + file)
                f = pd.DataFrame(f, columns=['index', 'item_recommendations', 'scores'])
                f = f.astype({'index': int})
                f = f[f['index'].isin(t_idx)]
                # f.sort_values(by=['index'])
            dataframes.append(f)
            dataframes_name.append(str(file))

    print('Extracting and check if the indices of submissions are the same')
    dataframes_idx = []
    for df in dataframes:
        idx = list(df['index'].values)
        dataframes_idx.append(idx)

    same_idx = []
    for i in range(len(dataframes_idx)):
        if i == 0:
            same_idx = set(dataframes_idx[i])
        else:
            same_idx = set(same_idx) & set(dataframes_idx[i])

    different_elems = set(t_idx) - set(same_idx)    # sono gli indici mancanti
    if len(different_elems) != 0:
        print(f'Missing {len(different_elems)} elements: {different_elems}')

    for i in range(len(dataframes)):
        mask = dataframes[i]['index'].isin(same_idx)
        dataframes[i] = dataframes[i][mask]

    print('Expand and create the dataset')
    df = pd.DataFrame()
    for i in range(len(dataframes)):
        if i == 0:
            df = convert_and_add_labels(dataframes[i], same_idx, name=dataframes_name[i])
        else:
            t = convert_and_assign_score(dataframes[i], dataframes_name[i])
            df = pd.merge(df, t, on=['index', 'item_id'])

    df = df.sort_values(by=['index', 'impression_position'])
    df.to_csv(save_path + '/' + name + '.csv', index=False)
    print('Dataset created. Columns are')
    print(df.columns)

if __name__ == '__main__':

    print('Local Test')
    # local train creation
    directory1 = ''
    create_dataset(mode='full', cluster='no_cluster', name='gbdt_test', directory=directory1, isTrain=False, save_path=directory1)

    print("Local Train")
    directory2 = ''
    create_dataset(mode='local', cluster='no_cluster', name='gbdt_train', directory=directory2, isTrain=False, save_path=directory2)



