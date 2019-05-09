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

            var = -1
            mean_time_per_step = -1

            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                x = x.loc[head_index.values[0]:clk.index.values[0]-1]

                if len(x) > 1:
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

                    if session_actions_num > 0:
                        var = round((var / session_actions_num) ** 0.5, 2)
                    else:
                        var = 0

            # return pd.Series({'mean_time_per_step':mean_time_per_step, 'frenzy_factor':var})
            return [(mean_time_per_step, var)]

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars=['user_id', 'session_id'], value_name='tuple').sort_values(
            by=['user_id', 'session_id']).dropna()
        s[['mean_time_per_step', 'frenzy_factor']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        s = s.drop(['variable', 'tuple'], axis=1)
        return s.reset_index(drop=True)

if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = FrenzyFactorSession(mode=mode, cluster='no_cluster')
    c.save_feature()
