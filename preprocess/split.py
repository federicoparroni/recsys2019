import pandas as pd

def split(df, perc_train=80):
    sorted_session_ids = df.groupby('session_id').first().sort_values('timestamp').reset_index()['session_id']
    slice_sorted_session_ids = sorted_session_ids.head(int(len(sorted_session_ids)*(perc_train/100)))
    df_train = df.loc[df['session_id'].isin(slice_sorted_session_ids)]
    df_test = df.loc[~df['session_id'].isin(slice_sorted_session_ids)]
    print('original: {} train: {} test: {}'.format(len(df), len(df_train), len(df_test)))
    df_train.to_csv('dataset/preprocessed/local_train.csv')
    df_test.to_csv('dataset/preprocessed/local_test.csv')

if __name__== "__main__":
    split(pd.read_csv('dataset/original/train.csv'))