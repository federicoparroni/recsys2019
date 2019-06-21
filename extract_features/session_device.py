from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
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
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        idxs_click = find_last_clickout_indices(df)
        tuple_list = []
        for i in idxs_click:
            user = df.at[i, 'user_id']
            sess = df.at[i, 'session_id']
            device = df.at[i, 'device']
            tuple_list.append((user, sess, device))
        return pd.DataFrame(tuple_list, columns=['user_id', 'session_id', 'session_device'])

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = SessionDevice(mode=mode, cluster=cluster)
    c.save_feature()
    c.read_feature(one_hot=True)
