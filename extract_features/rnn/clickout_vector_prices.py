import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import numpy as np
import pandas as pd
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from tqdm.auto import tqdm

class ClickoutVectorPrices(FeatureBase):

    """
    Extracts the prices vector of the clickout interactions
    | index | price_{i}
    price is an integer positive number, 0 <= i < 25
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'clickout_vector_prices'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()

        # find the clickout interactions
        res_df = df[['user_id','session_id','prices']]
        res_df = res_df[df.action_type == 'clickout item']

        # expand the prices as vector
        expanded_prices = res_df.prices.str.split('|', expand=True).fillna(0).astype('int')

        # scale log
        log_prices = np.log(expanded_prices +1)

        max_price = max(np.max(log_prices))
        min_price = min(np.min(log_prices))

        log_prices = (log_prices - min_price) / (max_price - min_price)

        # add the prices to the resulting df
        for i in range(25):
            res_df['price_{}'.format(i)] = log_prices.loc[:, i]
        
        return res_df.drop(['user_id','session_id','prices'], axis=1)


    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature()
        feature_cols = feature_df.columns
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df[feature_cols] = res_df[feature_cols].fillna(0)
        return res_df


if __name__ == '__main__':

    c = ClickoutVectorPrices()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
