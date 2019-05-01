from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import os
os.chdir("../")
print(os.getcwd())

class TimingFromLastInteractionImpression(FeatureBase):

    """
    how much time is elapsed and how many steps are passed from the last time a user
    interacted with an impression
    | user_id | session_id | item_id | seconds_from_last_interaction | steps_from_last_interaction
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'timing_from_last_interaction_impression'
        super(TimingFromLastInteractionImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            r = []
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0].split('|')
                x = x.loc[head_index.values[0]:clk.index.values[0]-1]
                df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][["reference", "action_type", "timestamp", "step"]]
                df_only_numeric = df_only_numeric.reset_index(drop=True)
                refs = list(df_only_numeric.reference.values)
                for i in impr:
                    if i in refs:
                        idx = len(refs) - refs[::-1].index(i) - 1
                        time = int(clk.timestamp) - int(df_only_numeric.loc[idx].timestamp)
                        step = int(clk.step) - int(df_only_numeric.loc[idx].step)
                        r.append((i, time, step))
                    else:
                        r.append((i, -1, -1))
            return r

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars = ['user_id', 'session_id'], value_name = 'tuple').sort_values(by=['user_id', 'session_id']).dropna()
        s[['item_id', 'seconds_from_last_interaction', 'steps_from_last_interaction']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        s = s.drop(['variable', 'tuple'], axis=1)
        s = s.reset_index(drop=True)
        return s
    
if __name__ == '__main__':
    c = TimingFromLastInteractionImpression(mode='small', cluster='no_cluster')
    c.save_feature()
