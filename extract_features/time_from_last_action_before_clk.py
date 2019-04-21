from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import os
os.chdir("../")
print(os.getcwd())

class TimePassedBeforeClickout(FeatureBase):

    """
    time passed before action before clickout. -1 if session consists of only an action
    | user_id | session_id | time_last_action_before_clk | frenzy_factor
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'time passed before clk'
        super(TimePassedBeforeClickout, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):

            if len(x) > 1:
                time_passed = int(x.tail().timestamp.values[1]) - int(x.tail().timestamp.values[0])
            else:
                time_passed = -1

            return time_passed

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        return pd.DataFrame({'user_id': [x[0] for x in s.index.values], 'session_id':[x[1] for x in s.index.values], 'time_passed_before_clk':[x for x in s.values]})

if __name__ == '__main__':
    c = TimePassedBeforeClickout(mode='small', cluster='no_cluster')
    c.save_feature()
