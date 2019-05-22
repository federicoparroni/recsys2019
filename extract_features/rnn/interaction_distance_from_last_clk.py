import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find

class InteractionDistanceFromLastClk(FeatureBase):

    """
    For each interaction lists the distance between the interaction and the last clickout of the session
    | index | distance (seconds)
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'interaction_distance_from_last_clk'

        super(InteractionDistanceFromLastClk, self).__init__(name=name, mode=mode, cluster=cluster, save_index=True)


    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        print('Sorting...')
        df = df.sort_values(['user_id','session_id','timestamp','step'])
        # find indices of last clickouts
        print('Finding last clickouts...')
        last_clickout_indices = find(df)
        # get only last clickout rows to use the timestamp column
        print('Getting only last clk dataframe...')
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','timestamp']]
        clickout_rows = clickout_rows.rename(columns={'timestamp':'clk_timestamp'})
        # add the timestamp of last clk for each session as column
        print('Getting tmp...')
        tmp_df = df[['user_id','session_id','step','action_type','timestamp']]
        tmp_df = pd.merge(tmp_df, clickout_rows, how='left', on=['user_id','session_id']).fillna(0)
        tmp_df.clk_timestamp = tmp_df.clk_timestamp.astype(int)

        # subtracts the timestamps, puts 0 if there is no clickout in the session
        def func(t, t_clko):
            res = np.empty(len(t))
            for i in tqdm(range(len(t))):
                if t_clko[i] == 0:
                    res[i] = 0
                else:
                    res[i] = t_clko[i] - t[i]
            return res
        print('Subtracting timestamps...')
        tmp_df['diff'] = func(tmp_df.timestamp.values, tmp_df.clk_timestamp)
        tmp_df['diff'] = tmp_df['diff'].astype(int)
        tmp_df['index'] = tmp_df.index
        feature = tmp_df[['index','diff']]

        return feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()

    c = InteractionDistanceFromLastClk(mode)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
