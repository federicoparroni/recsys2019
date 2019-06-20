import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import os
from extract_features.feature_base import FeatureBase
import data


class ChangeOfSortOrderBeforeCurrent(FeatureBase):

    """

    for every user_session group:
    the current sort order (if exists) on clk_out and the number of times a change of sort order has been
    clicked

    If no sort order is present in 'current_filters' sets default sort order to 'our recommendations'

    returns:
    user_id | session_id | current_sort_order | distance and recommended | ... | rating only

    current_sort_order corresponds to the sort order at time of a clk_out
    All (user_id, session_id) groups are present on the first two columns of the returned dataframe.

    The types of sorting orders reference and their unique correspondence to filters reference is the following:

    price only --> Sort by Price
    distance only --> Sort by Distance
    rating only --> Sort by Rating
    price and recommended --> Best Value
    rating and recommended --> Focus on Rating
    distance and recommended --> Focus on Distance
    our recommendations --> Sort by Popularity

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'changes_of_sort_order_before_current'
        super(ChangeOfSortOrderBeforeCurrent, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=[('current_sort_order', 'single')])

    def extract_feature(self):

        def func(x):

            def function_(d):
                df_filters_wout_last = d.iloc[:]
                return pd.get_dummies(df_filters_wout_last['reference']).sum(axis=0)

            def function_2(d):
                return pd.get_dummies(d['reference']).sum(axis=0)

            def return_current_sort(d):
                if len(d) > 0:
                    last = d.iloc[-1]
                    return last['reference']

            def return_sort_by_popularity(d):
                return pd.DataFrame({'current_sort_order': 'our recommendations'}, index=[0])

            tqdm.pandas()
            change_of_sort_order_train = x[(x['action_type'] == 'change of sort order') &
                                           (x['reference'] != 'interaction sort button')]

            if len(change_of_sort_order_train) == 0:  # non ci sono change of sort order

                return pd.DataFrame({'current_sort_order': 'our recommendations',
                                     'distance and recommended': 0,
                                     'distance only': 0,
                                     'our recommendations': 0,
                                     'price and recommended': 0,
                                     'price only': 0,
                                     'rating and recommended': 0,
                                     'rating only': 0}, index=[0])

            else:  # ci sono change of sort order

                index_of_corresponding_filter = change_of_sort_order_train.tail(1).index + 1
                current_filters = x.tail(1)['current_filters']

                if isinstance(x.tail(1)['current_filters'], str):  # se ci sono filtri nel clk

                    current_filters = x.tail(1)['current_filters'].split('|')

                possible_current_filter = x.loc[index_of_corresponding_filter]['reference']

                if possible_current_filter.isin(
                        current_filters).any():  # un change of sort order è dentro current_filter finale

                    if len(change_of_sort_order_train) == 1:  # se è l'unico change of sort order

                        r = pd.DataFrame({'current_sort_order': change_of_sort_order_train['reference'].values[0],
                                             'distance and recommended': 0,
                                             'distance only': 0,
                                             'our recommendations': 0,
                                             'price and recommended': 0,
                                             'price only': 0,
                                             'rating and recommended': 0,
                                             'rating only': 0}, index=[0])
                        r[change_of_sort_order_train['reference'].values[0]] += 1

                        return r

                    else:  # non è l'unico

                        feature = change_of_sort_order_train.groupby(['user_id', 'session_id']).apply(
                            function_).reset_index()
                        current_df = change_of_sort_order_train.groupby(['user_id', 'session_id']).apply(
                            return_current_sort).reset_index()
                        final_features = current_df.rename(columns={0: 'current_sort_order'}).merge(feature)

                        return final_features.drop(columns=[feature.columns[1], feature.columns[0]])

                else:  # nessun change of sort order è all'interno del current_filters finale

                    feature = change_of_sort_order_train.groupby(['user_id', 'session_id']).apply(
                        function_2).reset_index()
                    current_df = change_of_sort_order_train.groupby(['user_id', 'session_id']).apply(
                        return_sort_by_popularity).reset_index().drop(columns='level_2')
                    final_features = current_df.merge(feature)

                    return final_features.drop(columns=[feature.columns[1], feature.columns[0]])

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id','session_id']).progress_apply(func)
        cso = s.reset_index().drop(columns='level_2').fillna(0)
        cso = cso.astype(dtype={x: int for x in cso.columns.values[3:]})
        # s = s.drop(['current_sort_order'], axis=1)
        return cso


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection
    cluster = cluster_selection()
    mode = mode_selection()
    c = ChangeOfSortOrderBeforeCurrent(mode=mode, cluster=cluster)
    c.save_feature()
    #print(c.read_feature())
