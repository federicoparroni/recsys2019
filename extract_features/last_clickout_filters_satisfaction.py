import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
from preprocess_utils.last_clickout_indices import find
import data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm

class LastClickoutFiltersSatisfaction(FeatureBase):

    """
    Compute the percentage of impressions tags that satisfy the clickout active filters.
    | user_id | session_id | item_id | filter_sat
    filter_sat is a float number between 0 and 1, 0 <= i < 25
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'last_clickout_filters_satisfaction'
        columns_to_onehot = []

        super().__init__(name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot, save_index=False)


    def extract_feature(self):
        tqdm.pandas()

        tr = data.train_df(self.mode, cluster=self.cluster)
        te = data.test_df(self.mode, cluster=self.cluster)
        df = pd.concat([tr, te])
        accom_df = data.accomodations_one_hot()

        # this kind of filters are those of the type 'change-of-sort order'
        # they have a particular meaning and they must be handled in a separate feature
        change_sort_filters = set(['sort by price', 'sort by distance', 'sort by rating', 'sort by popularity',
                                    'focus on rating', 'focus on distance', 'best value'])

        # find the clickout rows
        last_clk = find(df)
        clickouts = df.loc[last_clk]
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
        satisfaction_percentage = []
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
                impression_satisfaction = list(np.sum( np.bitwise_and(filters_one_hot, impressions_features_one_hot), axis=1) / filters_len)
                satisfaction_percentage.append(impression_satisfaction)

            else:
                # there are only change-of-sort filters
                satisfaction_percentage.append(list(np.ones(len(impressions))))
            k += 1

        clickouts['satisfaction_percentage'] = satisfaction_percentage

        clickouts = clickouts.drop(['filters_list'], axis=1)

        expanded = pd.DataFrame({col: np.repeat(clickouts[col].values, clickouts.satisfaction_percentage.str.len()) \
                                for col in clickouts.columns.drop(['impress_list', 'satisfaction_percentage'])}).\
                                assign(**{'item_id': np.concatenate(clickouts.impress_list.values), \
                                        'satisfaction_percentage': np.concatenate(clickouts.satisfaction_percentage.values)})

        return expanded


    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature()
        feature_cols = feature_df.columns
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df[feature_cols] = res_df[feature_cols].fillna(0)
        return res_df


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = LastClickoutFiltersSatisfaction(mode=mode, cluster=cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()
