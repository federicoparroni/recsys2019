import pandas as pd
from tqdm.auto import tqdm
import time
tqdm.pandas()
import sys
import os
#os.chdir("/Users/Albo/Documents/GitHub/keyblade95/recsys2019")
from extract_features.feature_base import FeatureBase
import data


class SessionSortingFilters(FeatureBase):

    """

    for every user_session group:
    the current sort order (if exists) on clk_out and the number of times a different change of sort order has been
    clicked before current.

    returns:
    user_id | session_id | current_sort_order | sorting by_distance and recommended | ... | sorting by_rating only

    current_sort_order corresponds to the sort order at time of a clk_out
    All (user_id, session_id) groups are present on the first two columns of the returned dataframe.
    If a current_sort_order is not present is set to 0
    If a sort order before current is not present in session is set to zero.

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
        super(SessionSortingFilters, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):

            def function_(d):
               if len(d) > 1:
                    df_filters_wout_last = d.iloc[:-1]
                    return pd.get_dummies(df_filters_wout_last['reference']).sum(axis=0)

            def merge_rows(d):
                if d.shape[0] > 1:
                    return d.sum(axis=0, numeric_only=True)

            def return_current_sort(d):
                if len(d) > 0:
                    last = d.iloc[-1]
                    return last['reference']

            start = time.time()

            tqdm.pandas()
            change_of_sort_order_train = x[(x['action_type'] == 'change of sort order') &
                                               (x['reference'] != 'interaction sort button')]

            gbus_train = change_of_sort_order_train.groupby(['user_id', 'session_id'])

            feature = gbus_train.progress_apply(function_)

            f = pd.DataFrame(feature).reset_index()

            f_dummies = pd.get_dummies(f, columns=['level_2'], prefix='sorting by')

            for i in f_dummies.columns.values[3:]:
                f_dummies[i] = f_dummies[i] * f_dummies[0]

            f_dummies_dropped = f_dummies.drop(labels=[f_dummies.columns[2]], axis=1)

            f = f_dummies_dropped.groupby(['user_id', 'session_id']).progress_apply(merge_rows)

            last_df = f.reset_index()

            current_df = change_of_sort_order_train.groupby(['user_id', 'session_id']).progress_apply(return_current_sort)

            currentdf = pd.DataFrame(current_df).reset_index()

            final_features = currentdf.rename(columns={0: 'current_sort_order'}).merge(last_df)

            gtrain = x.groupby(['user_id', 'session_id'])

            keys = list(gtrain.groups.keys())

            df_train_grouped = pd.DataFrame({'user_id': [i[0] for i in keys], 'session_id': [i[1] for i in keys]})

            feature_final = df_train_grouped.merge(final_features, how='left').fillna(0)

            feature_final.astype(dtype={x: int for x in feature_final.columns.values[3:]})

            _time = time.time() - start
            elapsed = time.strftime('%Mm %Ss', time.gmtime(_time))
            print(f"elapsed on train local: {elapsed}")

            return feature_final

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = func(df)
        s = s.drop(['current_sort_order'], axis=1)
        return s


if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = SessionSortingFilters(mode=mode, cluster='no_cluster')
    c.save_feature()
    #print(c.read_feature())
