import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import numpy as np
import pandas as pd
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from tqdm.auto import tqdm

class ReferencePricePositionInLastClickout(FeatureBase):

    """
    Extracts the position of the reference price inside the last clickout impressions prices.
    If the reference is not present in the next clickout impressions, the position will be -1
    | index | price_pos
    price_pos is a number between 0-24 or -1
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'reference_price_position_in_last_clickout'
        columns_to_onehot = [('price_pos', 'single')]

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()
        # reset index to correct access
        df = df.sort_values(['user_id','session_id','timestamp','step']).reset_index()
        
        # find the last clickout rows
        last_clickout_idxs = find_last_clickout_indices(df)
        clickout_rows = df.loc[last_clickout_idxs, ['user_id','session_id','action_type','impressions','prices']]
        # cast the impressions and the prices to lists
        clickout_rows['impression_list'] = clickout_rows.impressions.str.split('|').apply(lambda x: list(map(int,x)))
        clickout_rows['price_list'] = clickout_rows.prices.str.split('|').apply(lambda x: list(map(int,x)))
        clickout_rows = clickout_rows.drop('impressions', axis=1)
        # order the prices lists
        clickout_rows['sorted_price_list'] = clickout_rows.price_list.apply(lambda x: sorted(x))
        clickout_rows = clickout_rows.drop('prices', axis=1)

        # find the interactions with numeric reference and not last clickouts
        reference_rows = df[['user_id','session_id','reference','action_type','index']]
        reference_rows = reference_rows[df.reference.str.isnumeric() == True].astype({'reference':'int'})
        # skip last clickouts
        reference_rows = reference_rows.loc[~reference_rows.index.isin(last_clickout_idxs)]
        reference_rows = reference_rows.drop('action_type', axis=1)
        ref_pos_series = np.ones(reference_rows.shape[0], dtype=int) * (-1)

        # iterate over the sorted reference_rows and clickout_rows
        j = 0
        clickout_indices = clickout_rows.index.values
        ckidx = clickout_indices[j]
        next_clickout_user_id = clickout_rows.at[ckidx, 'user_id']
        next_clickout_sess_id = clickout_rows.at[ckidx, 'session_id']
        k = 0
        for row in tqdm(zip(reference_rows.index, reference_rows.user_id, reference_rows.session_id, 
                            reference_rows.reference)):
            idx = row[0]
            # if the current index is over the last clickout, break
            if idx >= clickout_indices[-1]:
                break
            # find the next clickout index
            while idx > clickout_indices[j]:
                j += 1
                ckidx = clickout_indices[j]
                next_clickout_user_id = clickout_rows.at[ckidx, 'user_id']
                next_clickout_sess_id = clickout_rows.at[ckidx, 'session_id']
                next_clickout_impress = clickout_rows.at[ckidx, 'impression_list']
                next_clickout_prices = clickout_rows.at[ckidx, 'price_list']
                next_clickout_sortedprices = clickout_rows.at[ckidx, 'sorted_price_list']

            # check if row and next_clickout are in the same session
            if row[1] == next_clickout_user_id and row[2] == next_clickout_sess_id:
                try:
                    ref_idx = next_clickout_impress.index(row[3])
                    ref_price = int(next_clickout_prices[ref_idx])
                    ref_pos_series[k] = next_clickout_sortedprices.index(ref_price)
                except:
                    pass
            k += 1
        
        reference_rows['price_pos'] = ref_pos_series
        return reference_rows.drop(['user_id','session_id','reference'], axis=1).set_index('index')

    def post_loading(self, df):
        # drop the one-hot column -1, representing a non-numeric reference or a reference not present
        # in the clickout impressions
        if 'price_pos_-1' in df.columns:
            df = df.drop('price_pos_-1', axis=1)
        return df

    def join_to(self, df, one_hot=True):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature(one_hot=one_hot)
        feature_cols = feature_df.columns
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        if one_hot:
            # fill the non-joined NaN rows with 0
            res_df[feature_cols] = res_df[feature_cols].fillna(0).astype('int8')
        return res_df


if __name__ == '__main__':
    import utils.menu as menu

    c = ReferencePricePositionInLastClickout()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature(one_hot=True))
