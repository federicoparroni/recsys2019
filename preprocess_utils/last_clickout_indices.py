from tqdm.auto import tqdm

def find(df):
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