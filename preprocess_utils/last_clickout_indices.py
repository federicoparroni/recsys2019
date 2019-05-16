from tqdm.auto import tqdm
import pandas as pd
import numpy as np

def find(df, sort=True):
    """ 
        This assumes that the df is ordered by user_id, session_id, timestamp, step 
        However it is possible to specify if we want an explicit sorting by setting
        the flag sort to True
    """
    indices = []
    cur_ses = ''
    cur_user = ''
    if sort:
        df = df.sort_values(['user_id','session_id','timestamp','step'])
    temp_df = df[df.action_type == 'clickout item'][['user_id', 'session_id', 'action_type']]
    #temp_df = temp_df.sort_index()
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
    res_df = res_df.astype({'item_id': 'int'})

    return res_df