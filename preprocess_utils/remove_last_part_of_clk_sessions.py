import pandas as pd
from preprocess_utils.last_clickout_indices import find
from tqdm import tqdm

def remove_last_part_of_clk_sessions(df):
    """
    This function takes a dataframe and removes the interactions that
    occur after the last clickout of each session.
    """
    df = df.sort_values(by=['user_id','session_id','timestamp','step']).reset_index(drop=True)
    last_indices = find(df)
    last_clks = df.loc[last_indices]
    clks_sessions = last_clks.session_id.unique().tolist()
    clks_users = last_clks.user_id.unique().tolist()
    df_last_clks_sess_only = df[(df.session_id.isin(clks_sessions))&(df.user_id.isin(clks_users))][['user_id','session_id','action_type']]
    df_last_clks_sess_only_no_dupl = df_last_clks_sess_only.drop_duplicates(['user_id','session_id'])
    df_last_clks_sess_only_no_dupl['last_index'] = sorted(last_indices)
    df_last_clks_sess_only_no_dupl = df_last_clks_sess_only_no_dupl.drop('action_type',1)
    merged = pd.merge(df_last_clks_sess_only, df_last_clks_sess_only_no_dupl, how='left',on=['user_id','session_id']).set_index(df_last_clks_sess_only.index)
    indices_to_remove = []
    for t in tqdm(zip(merged.index, merged.last_index)):
        if t[0]>t[1]:
            indices_to_remove.append(t[0])
    return df.drop(indices_to_remove)
