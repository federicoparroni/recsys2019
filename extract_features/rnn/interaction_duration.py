import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
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
        def find(df):
            """
            Find Indices of last row of session
            """
            temp_df = df.copy()
            indices = []
            cur_ses = ''
            cur_user = ''
            for idx in tqdm(temp_df.index.values[::-1]):
                ruid = temp_df.at[idx, 'user_id']
                rsid = temp_df.at[idx, 'session_id']
                if (ruid != cur_user or rsid != cur_ses):
                    # append the original index
                    indices.append(idx)
                    cur_user = ruid
                    cur_ses = rsid
            return indices[::-1]

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        df = df.sort_values(['user_id','session_id','timestamp','step'])
        df['duration'] = df['timestamp'].shift(-1) - df['timestamp']

        last_row_session_indices = find(df)
        df = df[['user_id', 'session_id','step','action_type','timestamp','duration']]
        df.at[last_row_session_indices, 'duration'] = 0
        df['duration'] = df['duration'].astype(int)
        df['index'] = df.index

        return df[['index','duration']]

    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature().set_index('index')
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df['duration'] = res_df['duration'].fillna(0).astype('int')
        return res_df


if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()

    c = InteractionDuration(mode)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
