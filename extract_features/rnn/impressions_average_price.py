import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
import numpy as np

class ImpressionsAveragePrice(FeatureBase):

    """
    Compute the average price and standard deviation of the impressions.
    | item_id | prices_mean | prices_std
    
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'impressions_average_price'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', cluster='no_cluster', columns_to_onehot=columns_to_onehot)


    def extract_feature(self):
        df = data.full_df()

        # find the clickout rows
        clickout_rows = df[df.prices.notnull()][['impressions','prices']]
        # cast the impressions and the prices to lists
        clickout_rows['impressions'] = clickout_rows.impressions.str.split('|')
        clickout_rows['prices'] = clickout_rows.prices.str.split('|')

        clickout_rows = pd.DataFrame({col:np.concatenate(clickout_rows[col].values) \
                                    for col in clickout_rows.columns }).astype('int')
        # compute mean and standard deviation
        res_df = clickout_rows.groupby('impressions').agg(['mean','std']).reset_index()
        res_df.columns = ['_'.join(x) for x in res_df.columns.ravel()]
        res_df = res_df.rename(columns={'impressions_': 'item_id'})
        res_df['prices_std'] = res_df['prices_std'].fillna(0)
        
        return res_df

    
    def join_to(self, df):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature()
        # set the reference column to int
        res_df = df.copy()
        res_df.loc[(res_df.reference.str.isnumeric() != True) | (res_df.action_type == 'clickout item'), 'reference'] = 0
        res_df = res_df.astype({'reference': 'int'}).reset_index()
        res_df = res_df.merge(feature_df, how='left', left_on='reference', right_on='item_id').set_index('index')
        res_df = res_df.drop('item_id', axis=1)
        res_df['reference'] = df['reference']
        return res_df.fillna(0)


if __name__ == '__main__':
    import utils.menu as menu

    c = ImpressionsAveragePrice()
    
    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()
    
    print(c.read_feature())
