import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

class AveragePriceInNextClickout(FeatureBase):

    """
    Extracts the price of the impressions in the next clickout.
    If the reference is not present in the next clickout impressions, its price will be 0.
    | index | avg_price
    avg_price is a positive number
    """

    def __init__(self, mode, cluster):
        name = 'average_price_in_next_clickout'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()

        df = df.sort_index()
        # find the clickout rows
        clickout_rows = df[['user_id','session_id','action_type','prices']][df.action_type == 'clickout item']
        clickout_rows['avg_price'] = clickout_rows.prices.str.split('|').apply(lambda x: np.mean(list(map(int,x))))
        
        rows = df[['user_id','session_id']][df.reference.str.isnumeric() == True]
        rows['avg_price'] = 0.0

        # iterate over clickout_rows
        j = 0
        clickout_indices = clickout_rows.index.values
        for idx,row in tqdm(rows.iterrows()):
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
                    rows.at[idx,'avg_price'] = next_clickout.avg_price
                except:
                    pass
        
        return rows.drop(['user_id','session_id'], axis=1)


    def join_to(self, df, one_hot=True):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature()
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df['avg_price'] = res_df['avg_price'].fillna(0)
        return res_df


if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = AveragePriceInNextClickout(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
