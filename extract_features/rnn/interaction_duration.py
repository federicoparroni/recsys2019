import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm.auto import tqdm

class InteractionDuration(FeatureBase):

    """
    Extracts the duration of each interaction of the dataset
    | index | duration
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'interaction_duration'

        super(InteractionDuration, self).__init__(name=name, mode='full')


    def extract_feature(self):
        df = data.full_df()

        df = df.sort_values(['user_id','session_id','timestamp','step'])
        df['duration'] = df['timestamp'].shift(-1) - df['timestamp']

        last_row_session_indices = df.reset_index().groupby(['user_id','session_id'], as_index=False).last()['index'].values
        df = df[['user_id', 'session_id','step','action_type','timestamp','duration']]
        df.loc[last_row_session_indices, 'duration'] = 0
        df['duration'] = df['duration'].astype(int)
        df['index'] = df.index

        # scale log and min-max
        scaler = MinMaxScaler()
        df['duration'] = scaler.fit_transform( np.log(df['duration'].values+1).reshape((-1,1)) ).flatten()

        return df[['index','duration']]

    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature().set_index('index')
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df['duration'] = res_df['duration'].fillna(0) #.astype('int')
        return res_df


if __name__ == '__main__':
    import utils.menu as menu

    c = InteractionDuration()

    c.save_feature()

    print(c.read_feature())
