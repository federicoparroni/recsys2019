from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class SessionDevice(FeatureBase):

    """
    device used during a session:
    | user_id | session_id | session_device
    device is string
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'session_device'
        columns_to_onehot = [('session_device', 'single')]
        super(SessionDevice, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):

        def func(x):
            h = x.head(1)
            return h.device.values[0]

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        return pd.DataFrame({'user_id':[x[0] for x in s.index.values], 'session_id':[x[1] for x in s.index.values], 'session_device':[x[0] for x in s.values]})

if __name__ == '__main__':
    c = SessionDevice(mode='small', cluster='no_cluster')
    c.save_feature()
