from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class ChangeImpressionOrderPositionInSession(FeatureBase):
    """
    Features about the occurrence of an action 'search for poi' or 'change of sort order' or 'destination change' in the session.
    We compute 2 features per each occurrence:
    - first specifying the number of step from the first interaction (search_for_poi_distance_from_first_action and change_sort_order_distance_from_first_action)
    - one the number of steps from the clickout to predict (search_for_poi_distance_from_last_clickout, change_sort_order_distance_from_first_action)
    | user_id | session_id | search_for_poi_distance_from_last_clickout | search_for_poi_distance_from_first_action
    | change_sort_order_distance_from_last_clickout | change_sort_order_distance_from_first_action
    | destination_change_distance_from_last_clickout | destination_change_distance_from_first_action
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'change_impression_order_position_in_session'
        super(ChangeImpressionOrderPositionInSession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            y = x[x['action_type'] == 'clickout item']

            change_sort_order_distance_from_last_clickout = -1
            change_sort_order_distance_from_first_action = -1
            search_for_poi_distance_from_last_clickout = -1
            search_for_poi_distance_from_first_action = -1
            destination_change_distance_from_last_clickout = -1
            destination_change_distance_from_first_action = -1

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
                        search_for_poi_distance_from_first_action = last_poi_search_step - 1

                    sort_change_df = x[x.action_type == 'change of sort order']
                    if sort_change_df.shape[0] > 0:
                        sort_change_step = int(sort_change_df.tail(1).step.values[0])
                        change_sort_order_distance_from_last_clickout = int(
                            clk.step.values[0]) - sort_change_step
                        change_sort_order_distance_from_first_action = sort_change_step - 1

                    destination_change_df = x[x.action_type == 'search for destination']
                    if destination_change_df.shape[0] > 0:
                        destination_change_step = int(destination_change_df.tail(1).step.values[0])
                        destination_change_distance_from_last_clickout = int(
                            clk.step.values[0]) - destination_change_step
                        destination_change_distance_from_first_action = destination_change_step - 1

            return pd.Series({'search_for_poi_distance_from_last_clickout': search_for_poi_distance_from_last_clickout,
                              'search_for_poi_distance_from_first_action': search_for_poi_distance_from_first_action,
                              'change_sort_order_distance_from_last_clickout': change_sort_order_distance_from_last_clickout,
                              'change_sort_order_distance_from_first_action': change_sort_order_distance_from_first_action,
                              'destination_change_distance_from_last_clickout': destination_change_distance_from_last_clickout,
                              'destination_change_distance_from_first_action': destination_change_distance_from_first_action})

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)

        return s.reset_index()


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection
    cluster = cluster_selection()
    mode = mode_selection()
    c = ChangeImpressionOrderPositionInSession(mode=mode, cluster=cluster)
    c.save_feature()
