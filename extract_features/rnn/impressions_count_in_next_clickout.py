import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

class ImpressionsCountInNextClickout(FeatureBase):

    """
    Extracts the impressions count of the next clickout.
    | index | impr_count
    impr_count is a number between 0-24
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'impressions_count_in_next_clickout'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()

        df = df.sort_index()
        # find the clickout rows
        clickout_rows = df[['user_id','session_id','action_type','impressions']][df.action_type == 'clickout item']
        clickout_rows['impressions_count'] = clickout_rows.impressions.str.split('|').str.len()
        # prepare the resulting dataframe
        res_df = df[['user_id','session_id']].copy()
        res_df['impressions_count'] = 0

        # iterate over the sorted reference_rows and clickout_rows
        j = 0
        clickout_indices = clickout_rows.index.values

        ck_idx = clickout_indices[0]
        next_clickout_user_id = clickout_rows.at[ck_idx, 'user_id']
        next_clickout_sess_id = clickout_rows.at[ck_idx, 'session_id']
        for idx,row in tqdm(res_df.iterrows()):
            # if the current index is over the last clickout, break
            if idx > clickout_indices[-1]:
                break
            # find the next clickout index
            while idx > clickout_indices[j]:
                j += 1
                ck_idx = clickout_indices[j]
                next_clickout_user_id = clickout_rows.at[ck_idx, 'user_id']
                next_clickout_sess_id = clickout_rows.at[ck_idx, 'session_id']

            # check if row and next_clickout are in the same session
            if row.user_id == next_clickout_user_id and row.session_id == next_clickout_sess_id:
                res_df.at[idx,'impressions_count'] = clickout_rows.at[ck_idx, 'impressions_count']

        # create the 25 categories
        one_hot_counts = np.zeros((res_df.shape[0],25), dtype=np.int8)
        for i,c in enumerate(res_df.impressions_count.values):
            one_hot_counts[i, 0:c] = 1
        
        for i in range(25):
            res_df['impr_c{}'.format(i)] = one_hot_counts[:,i]

        return res_df.drop(['user_id','session_id','impressions_count'], axis=1)


    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature(one_hot=one_hot)
        feature_cols = feature_df.columns
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df[feature_cols] = res_df[feature_cols].fillna(0).astype('int8')
        return res_df


if __name__ == '__main__':
    import utils.menu as menu

    c = ImpressionsCountInNextClickout()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
