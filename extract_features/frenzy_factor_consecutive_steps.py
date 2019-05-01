from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

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
            y = x[x['action_type'] == 'clickout item']
            clk = y.tail(1)
            head_index = x.head(1).index

            # !! considering only the past for all non-test sessions! !!
            x = x.loc[head_index.values[0]:clk.index.values[0] - 1]

            if len(x) > 0:
                session_actions_num = int(clk.step.values[0])

                clickout_tm = int(clk.timestamp.values[0])
                time_length = clickout_tm - int(x.head(1).timestamp.values[0])

                mean_time_per_step = round(
                    time_length / (session_actions_num - 1), 2)

                var = 0
                prev_tm = 0

                for i in x.index:
                    curr_tm = int(x.at[i, 'timestamp'])

                    if prev_tm == 0:
                        prev_tm = curr_tm
                    else:
                        var += (mean_time_per_step - (curr_tm - prev_tm)) ** 2
                        prev_tm = curr_tm

                # summing var wrt of clickout
                var += (mean_time_per_step - (clickout_tm - prev_tm)) ** 2

                var = round((var / session_actions_num) ** 0.5, 2)
            else:
                var = -1
                mean_time_per_step = -1

            return mean_time_per_step, var

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)

        return pd.DataFrame({'user_id': [x[0] for x in s.index.values], 'session_id': [x[1] for x in s.index.values],
                             'mean_time_per_step': [x[0] for x in s.values], 'frenzy_factor': [x[1] for x in s.values]})

if __name__ == '__main__':
    c = FrenzyFactorSession(mode='full', cluster='cluster_sessions_no_numerical_reference')
    c.save_feature()
