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
            h = x.head(1)
            t = x.tail(1)
            ts = int(t.timestamp) - int(h.timestamp)
            return ts, t.step.values[0]

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        return pd.DataFrame({'user_id':[x[0] for x in s.index.values], 'session_id':[x[1] for x in s.index.values], 'length_timestamp':[x[0] for x in s.values], 'length_steps':[x[1] for x in s.values]})

if __name__ == '__main__':
    c = SessionLength(mode='small', cluster='no_cluster')
    c.save_feature()
