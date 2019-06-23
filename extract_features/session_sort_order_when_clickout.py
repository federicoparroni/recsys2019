from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class SessionSortOrderWhenClickout(FeatureBase):

    """
    device used during a session:
    | user_id | session_id | session_sort_order_when_clickout
    sort_order_active_when_clickout is string
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'session_sort_order_when_clickout'
        columns_to_onehot = [('sort_order_active_when_clickout', 'single')]
        super(SessionSortOrderWhenClickout, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):

        def func(x):
            change_of_sort_order_actions = x[x['action_type'] == 'change of sort order']
            if len(change_of_sort_order_actions) > 0:
                y = x[(x['action_type'] == 'clickout item')]
                if len(y) > 0:
                    clk = y.tail(1)
                    head_index = x.head(1).index
                    x = x.loc[head_index.values[0]:clk.index.values[0]-1]
                    change_of_sort_order_actions = x[x['action_type'] == 'change of sort order']
                    if len(change_of_sort_order_actions) > 0:
                        change_of_sort_order_actions = change_of_sort_order_actions.tail(1)
                        return change_of_sort_order_actions['reference'].values[0]
            return 'our recommendations'

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        return pd.DataFrame({'user_id':[x[0] for x in s.index.values], 'session_id':[x[1] for x in s.index.values], 'sort_order_active_when_clickout': s.values})

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = SessionSortOrderWhenClickout(mode=mode, cluster=cluster)
    c.save_feature()
