import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

class ReferencePriceInNextClickout(FeatureBase):

    """
    Extracts the position of the row reference inside the next clickout impressions.
    If the reference is not present in the next clickout impressions, its price will be 0.
    | index | user_id | session_id | price
    price is a positive number
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'reference_price_in_next_clickout'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()

        df = df.sort_index()
        # find the clickout rows
        clickout_rows = df[['user_id','session_id','action_type','impressions','prices']][df.action_type == 'clickout item']
        clickout_rows['impression_list'] = clickout_rows.impressions.str.split('|')
        clickout_rows['price_list'] = clickout_rows.prices.str.split('|')
        # find the interaction with numeric reference
        reference_rows = df[['user_id','session_id','reference','action_type']]
        reference_rows = reference_rows[df.reference.str.isnumeric() & (df.action_type != 'clickout item')]
        reference_rows = reference_rows.drop('action_type',axis=1)
        reference_rows['price'] = 0
        reference_rows = reference_rows.sort_index()

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
                    ref_idx = next_clickout.impression_list.index(row.reference)
                    reference_rows.at[idx, 'price'] = next_clickout.price_list[ref_idx]
                except:
                    pass
        
        return reference_rows.drop('reference', axis=1)


    def join_to(self, df, one_hot=True):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature().drop(['user_id','session_id'],axis=1)
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df['price'] = res_df.price.fillna(0).astype('int')
        return res_df


if __name__ == '__main__':

    c = ReferencePriceInNextClickout()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
