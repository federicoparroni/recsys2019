import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from extract_features.feature_base import FeatureBase
import data


class StepsBeforeLastClickout(FeatureBase):
    """

    for every session the last steps considering following heuristic:
    when a user clicks on some filter (whatever it is) or a new search is started, the page refreshes and the user
    is supposed to be scrolling and interacting with items from the beginning of the new refreshed list.
    If no filter or search is done, 'steps' corresponds to the length of session, otherwise it corresponds
    to the n of steps between the latest search or filter and the last clickout.

    user_id | session_id | session_length_timestamp | session_length_step

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'last_steps_before_clickout'
        super(StepsBeforeLastClickout, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):

            def last_important_steps(x):

                y = x[x.action_type == 'filter selection'].tail(1)
                i = x[x.action_type == 'search for item'].tail(1)
                d = x[x.action_type == 'search for destination'].tail(1)
                p = x[x.action_type == 'search for poi'].tail(1)
                steps = [y, i, d, p]
                _from = 1
                _from_serie = x.head(1)
                for i in steps:
                    if i.step.empty != True:
                        if i.step.values[0] > _from:
                            _from = i.step.values[0]
                            _from_serie = i
                return pd.Series({'session_length_timestamp': int(x.tail(1)['timestamp'].values[0]) -
                                                              int(_from_serie['timestamp'].values[0]),
                                  'session_length_step': int(x.tail(1).step) - int(_from) + 1})

            _important_steps = x.groupby(['user_id', 'session_id']).progress_apply(last_important_steps)
            return pd.DataFrame(_important_steps).reset_index()

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = func(df)
        return s


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = StepsBeforeLastClickout(mode=mode, cluster=cluster)
    c.save_feature()
    #print(c.read_feature())
