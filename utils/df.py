import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_dataframe(df, columns_to_scale, fill_nan=0):
    """ Return the dataframe with scaled columns """
    for col in columns_to_scale:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[col].fillna(value=fill_nan).values.reshape(-1, 1))
    return df


def ref_class_is_1(g):
    clickouts = g[g['clickout item'] == 1]
    return len(clickouts) > 0 and clickouts.iloc[-1]['ref_class'] == 1

def resample_sessions(df, by, when):
    """ Resample sessions that satisfy the specified condition.
    by (float):     percentage to resample the sessions (between 1 and 2)
    when (fn):      accepts a session as argument and return True or False to indicate if the session can be sampled
    Example function can be: utils.df.ref_class_is_1
    """
    assert 1 <= by <= 2
    by -= 1
    orig_len = len(df)
    print('Resampling sessions, ETA: {:.2f} minutes...'.format(orig_len * 0.0000025))

    temp_df = df.groupby(['user_id','session_id']).filter(lambda g: when(g) and np.random.rand() < by)
    temp_df.loc[:,'user_id'] = temp_df['user_id'] + '*'
    temp_df.loc[:,'session_id'] = temp_df['session_id'] + '*'
    
    # create new df
    new_len = orig_len + len(temp_df)
    print('Old length: {}\nNew length: {}'.format(orig_len, new_len))

    new_idx = -np.ones(new_len, dtype='int')
    new_idx[0:orig_len] = df.index.values
    res_df = pd.DataFrame(columns=df.columns, index=new_idx)
    res_df.iloc[0:orig_len,:] = df.values
    res_df.iloc[orig_len:,:] = temp_df.values
    return res_df.astype(df.dtypes)

