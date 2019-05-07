from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class SessionLength(FeatureBase):

    """
    length of a session:
    | user_id | session_id | session_length_timestamp | session_length_step
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'session_length'
        super(SessionLength, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0].split('|')
                x = x.loc[head_index.values[0]:clk.index.values[0]]
                h = x.head(1)
                t = x.tail(1)
                ts = int(t.timestamp) - int(h.timestamp)
                return pd.Series({'length_timestamp':ts, 'length_steps':t.step.values[0]})

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        return s.reset_index()

if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = SessionLength(mode=mode, cluster='no_cluster')
    c.save_feature()
