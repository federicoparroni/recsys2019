import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase

import data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

class ClickoutFiltersSatisfaction(FeatureBase):

    """
    Compute the percentage of impressions tags that satisfy the clickout active filters.
    | index | filter_sat_{i}
    filter_sat_{i} is a float number between 0 and 1, 0 <= i < 25
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'clickout_filters_satisfaction'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()
        #df = data.train_df('small')
        accom_df = data.accomodations_one_hot()

        # this kind of filters are those of the type 'change-of-sort order'
        # they have a particular meaning and they must be handled in a separate feature
        change_sort_filters = set(['sort by price', 'sort by distance', 'sort by rating', 'sort by popularity',
                                    'focus on rating', 'focus on distance', 'best value'])
        
        # find the clickout rows
        clickouts = df[(df.action_type == 'clickout item')] # & df.current_filters.notnull()]
        clickouts = clickouts[['user_id','session_id','current_filters','impressions']]
        # split the filters and the impressions
        clickouts['filters_list'] = clickouts['current_filters'].str.lower().str.split('|').fillna('')
        clickouts['impress_list'] = clickouts['impressions'].str.split('|')
        clickouts = clickouts.drop(['impressions','current_filters'], axis=1)
        # cast the impressions to int
        clickouts['impress_list'] = clickouts['impress_list'].apply(lambda x: list(map(int, x)))
        
        # create the binarizer with the same classes as the accomodations one-hot
        mlb = MultiLabelBinarizer(classes=accom_df.columns.str.lower())
        
        # iterate over the clickouts and one-hot
        print('Total interactions:', clickouts.shape[0])
        result = np.zeros((clickouts.shape[0],25), dtype='float')
        k = 0
        for idx in tqdm(clickouts.index):
            filters = clickouts.at[idx, 'filters_list']
            impressions = clickouts.at[idx, 'impress_list']
            # do not consider change-of-sort filters
            filters = set(filters).difference(change_sort_filters)
            # fix some wrong filters names
            if 'gay friendly' in filters:
                filters.remove('gay friendly')
                filters.add('gay-friendly')
            if 'internet (rooms)' in filters:
                filters.remove('internet (rooms)')
                filters.add('free wifi (rooms)')
            filters = list(filters)
            filters_len = len(filters)
            if filters_len > 0:
                # one-hot the filters
                filters_one_hot = mlb.fit_transform([filters])[0]
                # take the one-hot of the impressions tags
                impressions_features_one_hot = accom_df.loc[impressions].values

                satisfaction_percentage = np.sum( np.bitwise_and(filters_one_hot, impressions_features_one_hot), axis=1) / filters_len
                result[k, 0:len(impressions)] = satisfaction_percentage
            else:
                # there are only change-of-sort filters
                result[k, 0:len(impressions)] = 1
            k += 1
        
        result = result.round(4)

        # add the 25 new columns
        for i in range(25):
            clickouts['satisf_perc_{}'.format(i)] = result[:,i]

        return clickouts.drop(['user_id','session_id','filters_list','impress_list'], axis=1)


    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature()
        feature_cols = feature_df.columns
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df[feature_cols] = res_df[feature_cols].fillna(0)
        return res_df


if __name__ == '__main__':

    c = ClickoutFiltersSatisfaction()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
