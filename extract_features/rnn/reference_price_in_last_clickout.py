import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from tqdm.auto import tqdm

class ReferencePriceInLastClickout(FeatureBase):

    """
    Extracts the position of the row reference inside the last clickout impressions.
    If the reference is not present in the last clickout impressions, its price will be 0.
    | index | price
    price is a positive number
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'reference_price_in_last_clickout'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()
        df = df.sort_values(['user_id','session_id','timestamp','step']).reset_index()

        # find the last clickout rows
        last_clickout_idxs = find_last_clickout_indices(df)
        clickout_rows = df.loc[last_clickout_idxs, ['user_id','session_id','action_type','impressions','prices']]
        clickout_rows['impression_list'] = clickout_rows.impressions.str.split('|')
        clickout_rows['price_list'] = clickout_rows.prices.str.split('|')
        # find the interactions with numeric reference
        reference_rows = df[['user_id','session_id','reference','action_type', 'index']]
        reference_rows = reference_rows[df.reference.str.isnumeric() == True]
        # skip last clickouts
        reference_rows = reference_rows.loc[~reference_rows.index.isin(last_clickout_idxs)]
        reference_rows = reference_rows.drop('action_type',axis=1)
        reference_rows['price'] = 0

        # iterate over the sorted reference_rows and clickout_rows
        j = 0
        clickout_indices = clickout_rows.index.values
        ckidx = clickout_indices[j]
        next_clickout_user_id = clickout_rows.at[ckidx, 'user_id']
        next_clickout_sess_id = clickout_rows.at[ckidx, 'session_id']
        for idx,row in tqdm(reference_rows.iterrows()):
            # if the current index is over the last clickout, break
            if idx >= clickout_indices[-1]:
                break
            # find the next clickout index
            while idx > clickout_indices[j]:
                j += 1
                ckidx = clickout_indices[j]
                next_clickout_user_id = clickout_rows.at[ckidx, 'user_id']
                next_clickout_sess_id = clickout_rows.at[ckidx, 'session_id']

            # check if row and next_clickout are in the same session
            if row.user_id == next_clickout_user_id and row.session_id == next_clickout_sess_id:
                try:
                    ref_idx = clickout_rows.at[ckidx, 'impression_list'].index(row.reference)
                    reference_rows.at[idx, 'price'] = clickout_rows.at[ckidx, 'price_list'][ref_idx]
                except:
                    pass
        
        return reference_rows.drop(['user_id','session_id','reference'], axis=1).set_index('index')


    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature()
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df['price'] = res_df.price.fillna(0).astype('int')
        return res_df


if __name__ == '__main__':

    c = ReferencePriceInLastClickout()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
