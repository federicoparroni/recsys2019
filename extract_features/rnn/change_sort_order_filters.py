import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase

import data
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

class ChangeSortOrderFilters(FeatureBase):

    """
    Encode the change-of-sort-order filters of the individual interactions.
    | index | sort_rating | sort_pop | sort_price
    The columns have 1 or 0 as values
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'change_sort_order_filters'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()

        # mapping and encoding
        change_sort_filters = set(['sort by price','best value', 
                           'sort by rating','focus on rating',
                           'sort by popularity'])
        sof_classes = ['sort_rating', 'sort_pop', 'sort_price']
        mapping = {
            'sort by price':      [0,0,1],
            'best value':         [0,1,1],
            'sort by rating':     [1,0,0],
            'focus on rating':    [1,1,0],
            'sort by popularity': [0,1,0],
        }

        rows = df[(df.action_type == 'clickout item') & df.current_filters.notnull()]
        rows = rows[['current_filters']]
        # filter the filters by the sort filters and re-cast to list
        rows['filters_list'] = rows['current_filters'].str.lower().str.split('|')\
                                        .progress_apply(lambda x: list(set(x) & change_sort_filters))
        rows = rows.drop(['current_filters'], axis=1)
        rows = rows[rows['filters_list'].str.len() > 0]
        rows['filters_list'] = rows['filters_list'].apply(lambda x: x[0])

        # iterate over the interactions
        print('Total interactions:', rows.shape[0])
        matrix = np.zeros((rows.shape[0], len(sof_classes)), dtype='int8')
        k = 0
        for fl in tqdm(rows['filters_list'].values):
            matrix[k,:] = mapping[fl]
            k += 1

        # add the 3 new columns
        for i,col_name in enumerate(sof_classes):
            rows[col_name] = matrix[:,i]

        return rows.drop('filters_list', axis=1)


    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature()
        feature_cols = feature_df.columns
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df[feature_cols] = res_df[feature_cols].fillna(0).astype('int8')
        return res_df


if __name__ == '__main__':

    c = ChangeSortOrderFilters()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
