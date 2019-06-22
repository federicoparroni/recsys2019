from extract_features.feature_base import FeatureBase
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
import data
import pandas as pd
import numpy as np
from tqdm import tqdm


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

        tr = data.train_df(mode=self.mode, cluster=self.cluster)
        te = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([tr, te])
        idxs = sorted(find(df))
        means = []
        stds = []
        for i in tqdm(idxs):
            a_user = df.at[i, 'user_id']
            a_sess = df.at[i, 'session_id']
            a_time = df.at[i, 'timestamp']
            j = i-1
            diffs = []
            while j >= 0:
                try:
                    new_user = df.at[j, 'user_id']
                    new_sess = df.at[j, 'session_id']
                    new_time = df.at[j, 'timestamp']
                    if new_user == a_user and new_sess == a_sess:
                        diffs.append(a_time - new_time)
                    else:
                        break
                    j -= 1
                    a_time = new_time
                except:
                    j -= 1
            if len(diffs) > 0:
                np_diffs = np.array(diffs)
                means.append(np.mean(np_diffs))
                stds.append(np.std(np_diffs))
            else:
                means.append(-1)
                stds.append(-1)

        total = df.loc[idxs, ['user_id', 'session_id']]
        total['mean_time_per_step'] = means
        total['frenzy_factor'] = stds
        return total.reset_index(drop=True)


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = FrenzyFactorSession(mode=mode, cluster=cluster)
    c.save_feature()
