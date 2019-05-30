import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
import numpy as np
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from tqdm.auto import tqdm

class SessionsImpressionsCount(FeatureBase):

    """
    Extracts the impressions count of the next clickout.
    The count is multi-hot from 0 to n (where n is the session impression count)
    | index | impr_c{i}
    i is a number between 0-24
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'session_impressions_count'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=False)



    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()
        df = df.sort_values(['user_id','session_id','timestamp','step']).reset_index()
        
        # find the last clickout rows
        last_clickout_idxs = find_last_clickout_indices(df)
        clickout_rows = df.loc[last_clickout_idxs, ['user_id','session_id','impressions','index']]
        clickout_rows['impressions_count'] = clickout_rows.impressions.str.split('|').str.len()
        clickout_rows = clickout_rows.drop('impressions', axis=1)

        # multi-hot the counts
        one_hot_counts = np.zeros((clickout_rows.shape[0],25), dtype=np.int8)
        for i,c in tqdm(enumerate(clickout_rows.impressions_count.values)):
            one_hot_counts[i, 0:c] = 1
        
        # add to the clickouts
        for i in range(25):
            clickout_rows['impr_c{}'.format(i)] = one_hot_counts[:,i]
        
        return clickout_rows.drop('impressions_count', axis=1).set_index('index')


    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature(one_hot=one_hot)
        feature_cols = feature_df.drop(['user_id','session_id'], axis=1).columns
        res_df = df.reset_index().merge(feature_df, how='left').set_index('index')
        res_df[feature_cols] = res_df[feature_cols].fillna(0).astype('int8')
        return res_df


if __name__ == '__main__':
    import utils.menu as menu

    c = SessionsImpressionsCount()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
