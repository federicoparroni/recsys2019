import pandas as pd
import numpy as np

def split(df, perc_train=80):
    # train-test split, as they did
    sorted_session_ids = df.groupby('session_id').first().sort_values('timestamp').reset_index()['session_id']
    slice_sorted_session_ids = sorted_session_ids.head(int(len(sorted_session_ids)*(perc_train/100)))
    df_train = df.loc[df['session_id'].isin(slice_sorted_session_ids)]
    df_test = df.loc[~df['session_id'].isin(slice_sorted_session_ids)]

    # remove clickout from test and save an handle
    groups = df_test[df_test['action_type'] == 'clickout item'].groupby('user_id', as_index=False)
    remove_reference_tuples = groups.apply(lambda x: x.sort_values(by=['timestamp'], ascending=True).tail(1))
    df_handle = df.loc[[e[1] for e in remove_reference_tuples.index.tolist()],['user_id', 'session_id', 'timestamp', 'step', 'reference', 'impressions']]
    for e in remove_reference_tuples.index.tolist():
        df_test.at[e[1], 'reference'] = np.nan
    
    # save them all
    df_train.to_csv('dataset/preprocessed/local_train.csv')
    df_test.to_csv('dataset/preprocessed/local_test.csv')
    df_handle.to_csv('dataset/preprocessed/local_handle.csv')

if __name__== "__main__":
    split(pd.read_csv('dataset/original/train.csv'))
