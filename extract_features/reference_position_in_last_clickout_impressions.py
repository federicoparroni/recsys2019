import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase

import data
import pandas as pd
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from tqdm.auto import tqdm

class ReferencePositionInLastClickoutImpressions(FeatureBase):

    """
    Extracts the position of the row reference inside the last clickout impressions.
    If the reference is not present in the last clickout impressions, the position will be -1
    | index | user_id | session_id | ref_pos
    ref_pos is a number between 0-24 or -1
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'reference_position_in_last_clickout_impressions'
        columns_to_onehot = [('ref_pos', 'single')]

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)
        self.one_hot_prefix = 'rp'


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()

        df = df.sort_index()
        # find the clickout rows
        last_clickout_idxs = find_last_clickout_indices(df)
        clickout_rows = df.loc[last_clickout_idxs, ['user_id','session_id','action_type','impressions']]
        clickout_rows['impression_list'] = clickout_rows.impressions.str.split('|')
        clickout_rows = clickout_rows.drop('impressions', axis=1)

        # find the interaction with numeric reference
        reference_rows = df[['user_id','session_id','reference','action_type']]
        reference_rows = reference_rows[df.reference.str.isnumeric() & (df.action_type != 'clickout item')]
        reference_rows = reference_rows.drop('action_type', axis=1)
        reference_rows['ref_pos'] = -1

        # iterate over the sorted reference_rows and clickout_rows
        j = 0
        clickout_indices = clickout_rows.index.values
        for idx,row in tqdm(reference_rows.iterrows()):
            # if the current index is over the last clickout, break
            if idx >= clickout_indices[-1]:
                break
            # find the next clickout index
            while idx > clickout_indices[j]:
                j += 1
            next_clickout = clickout_rows.loc[clickout_indices[j]]

            # check if row and next_clickout are in the same session
            if row.user_id == next_clickout.user_id and row.session_id == next_clickout.session_id:
                try:
                    reference_rows.at[idx,'ref_pos'] = next_clickout.impression_list.index(row.reference)
                except:
                    pass
        
        return reference_rows.drop('reference', axis=1)

    def post_loading(self, df):
        # drop the one-hot column -1, representing a non-numeric reference or a reference not present
        # in the clickout impressions
        if 'rp_-1' in df.columns:
            df = df.drop('rp_-1', axis=1)
        return df

    def join_to(self, df, one_hot=True):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature(one_hot=one_hot).drop(['user_id','session_id'],axis=1)
        feature_cols = feature_df.columns
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        if one_hot:
            # fill the non-joined NaN rows with 0
            res_df[feature_cols] = res_df[feature_cols].fillna(0).astype('int8')
        return res_df


if __name__ == '__main__':

    c = ReferencePositionInLastClickoutImpressions()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature(one_hot=True))
