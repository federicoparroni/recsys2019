from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import os
#os.chdir("../")
#print(os.getcwd())

class FrenzyFactorSession(FeatureBase):

    """
    mean of time passed during consecutive steps and squared variance between every 2 consecutive steps in ms (frenzy factor of a user)
    | user_id | session_id | mean_time_per_step | frenzy_factor

    WARNING:
    Since it considers only consecutive steps, frenzy factor works ONLY with non-filtered
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'frenzy_factor_session'
        super(FrenzyFactorSession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):

            if len(x) > 1:
                session_actions_num = int(x.tail(1).step)

                time_length = int(x.tail(1).timestamp) - int(x.head(1).timestamp)

                mean_time_per_step = round(time_length / session_actions_num , 2)

                var = 0
                prev_tm = 0
                for i, row in x.iterrows():
                    if prev_tm == 0:
                        prev_tm = int(row.timestamp)
                    else:
                        var += (mean_time_per_step - (int(row.timestamp) - prev_tm) )**2
                        prev_tm = int(row.timestamp)

                var = round((var/session_actions_num)**0.5, 2)
            else:
                var = -1
                mean_time_per_step = -1

            return mean_time_per_step, var

        #train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df =test# pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)

        return pd.DataFrame({'user_id': [x[0] for x in s.index.values], 'session_id': [x[1] for x in s.index.values],
                             'mean_time_per_step': [x[0] for x in s.values], 'frenzy_factor': [x[1] for x in s.values]})

if __name__ == '__main__':
    c = FrenzyFactorSession(mode='full', cluster='cluster_sessions_no_numerical_reference')
    c.save_feature()
