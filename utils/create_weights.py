import pandas as pd
import numpy as np
import math
import os
import data
from tqdm import tqdm

def create_weights_position(train_df, mode,cluster):
    train = data.train_df(mode, cluster)
    test = data.test_df(mode, cluster)
    df = pd.concat([train, test])
    # get for each user-session the position of the clicked item
    df_clks = df[(df['reference'].str.isnumeric()==True)&(df['action_type']=='clickout item')][['user_id','session_id','reference','impressions']]
    df_clks.impressions = df_clks.impressions.str.split('|')
    new_col = []
    for t in tqdm(zip(df_clks.reference, df_clks.impressions)):
        if t[0] in t[1]:
            new_col.append(t[1].index(t[0])+1)
        else:
            new_col.append(-1)
    df_clks['pos_clicked'] = new_col
    pos_clicked_list = df_clks.pos_clicked.tolist()
    # create dictionary {pos:score}
    dict_pos_score = {}
    for i in tqdm(range(1,26)):
        dict_pos_score[i] = 1-(pos_clicked_list.count(i)/len(pos_clicked_list)) # the function is 1-(#pos/tot_rowså)
    # group per user-session
    group = train_df.drop_duplicates(['user_id','session_id'])[['user_id','session_id']].reset_index(drop=True)
    # assign weight
    gr = train_df[train_df.label==1][['user_id','session_id','impression_position']]
    new_col = []
    for p in gr.impression_position:
        if p not in range(1,26):
            new_col.append(0)
        else:
            new_col.append(dict_pos_score[p])
    gr['weight'] = new_col
    final = pd.merge(group, gr, how='left', on=['user_id','session_id']).fillna(0)
    sample_weights = final['weight'].values
    return sample_weights

def create_balanced_weights_position(train_df, mode,cluster):
    train = data.train_df(mode, cluster)
    test = data.test_df(mode, cluster)
    df = pd.concat([train, test])
    # get for each user-session the position of the clicked item
    df_clks = df[(df['reference'].str.isnumeric()==True)&(df['action_type']=='clickout item')][['user_id','session_id','reference','impressions']]
    df_clks.impressions = df_clks.impressions.str.split('|')
    new_col = []
    for t in tqdm(zip(df_clks.reference, df_clks.impressions)):
        if t[0] in t[1]:
            new_col.append(t[1].index(t[0])+1)
        else:
            new_col.append(-1)
    df_clks['pos_clicked'] = new_col
    pos_clicked_list = df_clks.pos_clicked.tolist()
    # get the number of occurences of each class
    dict_pos_support = {}
    for i in tqdm(range(1,26)):
        dict_pos_support[i] = pos_clicked_list.count(i)
    # create dictionary {pos:score}
    min_support = dict_pos_support[min(dict_pos_support, key=dict_pos_support.get)]
    dict_pos_scores = {}
    for i in dict_pos_support:
        dict_pos_scores[i] = min_support / dict_pos_support[i]
    # group per user-session
    group = train_df.drop_duplicates(['user_id','session_id'])[['user_id','session_id']].reset_index(drop=True)
    # assign weight
    gr = train_df[train_df.label==1][['user_id','session_id','impression_position']]
    new_col = []
    for p in gr.impression_position:
        if p not in range(1,26):
            new_col.append(0)
        else:
            new_col.append(dict_pos_scores[p])
    gr['weight'] = new_col
    final = pd.merge(group, gr, how='left', on=['user_id','session_id']).fillna(0)
    sample_weights = final['weight'].values
    return sample_weights

def create_log_weights(train_df):
    d = {}
    for i in range(1,26):
        d[i]=math.sqrt(math.log(1+i, 26))
    # group per user-session
    group = train_df.drop_duplicates(['user_id','session_id'])[['user_id','session_id']].reset_index(drop=True)
    # assign weight
    gr = train_df[train_df.label==1][['user_id','session_id','impression_position']]
    new_col = []
    for p in gr.impression_position:
        if p not in range(1,26):
            new_col.append(0)
        else:
            new_col.append(d[p])
    gr['weight'] = new_col
    final = pd.merge(group, gr, how='left', on=['user_id','session_id']).fillna(0)
    sample_weights = final['weight'].values
    return sample_weights

if __name__ == "__main__":
    from utils.menu import mode_selection
    from utils.menu import cluster_selection
    mode = mode_selection()
    cluster = cluster_selection()
    kind = input('insert the kind: ')

    folder = f'dataset/preprocessed/{cluster}/{mode}/xgboost/{kind}/'
    train_df = pd.read_csv(os.path.join(folder, 'train_df.csv'))

    balanced_w = create_balanced_weights_position(train_df, mode,cluster)
    np.save(os.path.join(folder, 'balanced_weights'), balanced_w)
    print('balanced_weights saved')
