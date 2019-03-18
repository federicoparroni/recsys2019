import pandas as pd
import data
import numpy as np
from tqdm import tqdm
import utils.check_folder as cf

def merge_consecutive_equal_actions():
    tqdm.pandas()
    test = data.test_df('full')
    test_grouped_by_session_id = test.groupby('session_id')
    merged = test_grouped_by_session_id.progress_apply(_merge_consecutive_equal_actions)
    cf.check_folder('dataset/cleaned_csv')
    merged.to_csv('dataset/cleaned_csv/test.csv')

def _merge_consecutive_equal_actions(df):
    df_cleaned = pd.DataFrame(columns=['user_id', 'session_id', 'timestamp', 'step', 'action_type',
                                       'occurences', 'reference', 'platform', 'city', 'device', 'current_filters',
                                       'impressions', 'prices'])
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if i == 0:
            row['occurences'] = 1
            df_cleaned = df_cleaned.append(row, ignore_index=True)
        else:
            a_r = np.array(row[['action_type', 'reference']])
            last = np.array(df_cleaned.tail(1)[['action_type', 'reference']])
            if (a_r == last).all():
                df_cleaned.at[df_cleaned.tail(1).index[0], 'occurences'] += 1
                df_cleaned.tail(1)[['timestamp', 'step']] = row[['timestamp', 'step']].values
            else:
                row['occurences'] = 1
                df_cleaned = df_cleaned.append(row)
    return df_cleaned



if __name__ == '__main__':
    merge_consecutive_equal_actions()