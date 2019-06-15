from tqdm.auto import tqdm
import pandas as pd
import numpy as np


def find(df):
    """ 
    Return the last clickouts of each session in df.
    """
    temp_df = df.sort_values(['user_id','session_id','timestamp','step'])
    if 'index' in temp_df.columns:
        temp_df = temp_df.rename(columns={'index':'$_index'})

    temp_df = temp_df.reset_index()
    temp_df = temp_df[temp_df.action_type == 'clickout item'][['index','user_id','session_id','action_type','reference']]

    indices = []
    cur_ses = ''
    cur_user = ''
    for idx in tqdm(temp_df.index.values[::-1]):
        #print(temp_df.at[idx, 'index'])
        ruid = temp_df.at[idx, 'user_id']
        rsid = temp_df.at[idx, 'session_id']
        reference = temp_df.at[idx, 'reference']
        if (ruid != cur_user or rsid != cur_ses):
            # append the original index
            indices.append(temp_df.at[idx, 'index'])
            cur_user = ruid
            cur_ses = rsid
        else:
            if pd.isnull(reference):
                indices = indices[:-1]
                indices.append(temp_df.at[idx, 'index'])
    return indices[::-1]


# def find(df):
#     """
#     Return the last clickouts of each session in df.
#     """
#     df = df.sort_values(['user_id','session_id','timestamp','step'])
#     clickout_df = df[(df.action_type == "clickout item")]
#     mask = clickout_df.reference.isnull()
#     clickout_sessions = list(set(clickout_df[mask].session_id))
#     clickout_indices = list(set(clickout_df[mask].index))
#
#     clickout_df = clickout_df.drop_duplicates("session_id", keep="last")
#     clickout_df = clickout_df[~clickout_df.session_id.isin(clickout_sessions)]
#     i = list(set(list(clickout_df.index) + clickout_indices))
#     i.sort()
#     return i

def expand_impressions(df):
    res_df = df.copy()
    res_df.impressions = res_df.impressions.str.split('|')
    res_df = res_df.reset_index()

    res_df = pd.DataFrame({
        col: np.repeat(res_df[col].values, res_df.impressions.str.len())
        for col in res_df.columns.drop('impressions')}
    ).assign(**{'impressions': np.concatenate(res_df.impressions.values)})[res_df.columns]

    res_df = res_df.rename(columns={'impressions': 'item_id'})
    res_df = res_df.astype({'item_id': 'int'})

    return res_df
