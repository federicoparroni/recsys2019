from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class ChangeOrderImpression(FeatureBase):

    """
    Avg position of the item clicked AND interacted by the user with respect to the price position sorted by the cheapest
    during the session sorted by price ascendent.
    -1 if no other interaction is present.
    Position of the impressions interacted usually when info about impression is available. (number from 1 to 25)
    also position of the last impression interacted/clicked (this hopes to let apply what lazy user recommender does)
    -1 is not available.
    | user_id | session_id | mean_price_interacted | mean_cheap_pos_interacted | 'mean_pos' | 'pos_last_reference'
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'change_impression_order_position_in_session.py'
        super(ChangeOrderImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            y = x[x['action_type'] == 'clickout item']

            change_sort_order_distance_from_last_clickout = -1
            change_sort_order_distance_from_first_action = -1
            search_for_poi_distance_from_last_clickout = -1
            search_for_poi_distance_from_first_action = -1

            if len(y) > 0:
                clk = y.tail(1)
                head_df = x.head(1)
                head_index = head_df.index
                x = x.loc[head_index.values[0]:clk.index.values[0] - 1]

                if len(x) > 1:
                    clk = x.tail(1)

                    poi_search_df = x[x.action_type == 'search for poi']
                    if poi_search_df.shape[0] > 0:
                        last_poi_search_step = int(
                            poi_search_df.tail(1).step.values[0])
                        search_for_poi_distance_from_last_clickout = int(
                            clk.step.values[0]) - last_poi_search_step
                        search_for_poi_distance_from_first_action = last_poi_search_step - int(
                            head_df.step.values[0])

                    sort_change_df = x[x.action_type == 'change of sort order']
                    if sort_change_df.shape[0] > 0:
                        sort_change_step = int(sort_change_df.tail(1).step.values[0])
                        change_sort_order_distance_from_last_clickout = int(
                            clk.step.values[0]) - sort_change_step
                        change_sort_order_distance_from_first_action = sort_change_step - int(
                            head_df.step.values[0])


            return pd.Series({'search_for_poi_distance_from_last_clickout': search_for_poi_distance_from_last_clickout, 'search_for_poi_distance_from_first_action':search_for_poi_distance_from_first_action,
                             'change_sort_order_distance_from_last_clickout': change_sort_order_distance_from_last_clickout, 'change_sort_order_distance_from_first_action': change_sort_order_distance_from_first_action})

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)

        return s.reset_index()
if __name__ == '__main__':
    c = ChangeOrderImpression(mode='small', cluster='no_cluster')
    c.save_feature()
