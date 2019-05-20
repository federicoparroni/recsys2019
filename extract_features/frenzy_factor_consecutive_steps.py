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

        train = data.train_df(mode=self.mode)
        test = data.test_df(mode=self.mode)
        df_full = pd.concat([train, test])[['user_id', 'session_id', 'timestamp', 'step', 'action_type','reference']]
        last_clickout_indices = sorted(find(df_full), reverse=True)
        df = df_full[['user_id', 'session_id', 'timestamp']]
        cuid = df.at[last_clickout_indices[0], 'user_id']
        csid = df.at[last_clickout_indices[0], 'session_id']
        i = last_clickout_indices[0]
        means = []
        variances = [] 
        for idx in tqdm(last_clickout_indices):
            
            cuid = df.at[idx, 'user_id']
            csid = df.at[idx, 'session_id']
            time_differences = np.array([])
            i = idx - 1
            while df.at[i, 'user_id'] == cuid and df.at[i, 'session_id'] == csid:
                time_differences = np.append(time_differences, [df.at[i+1, 'timestamp'] - df.at[i, 'timestamp']])
                i -= 1
                if i == -1:
                    break
            means.append(np.mean(time_differences))
            variances.append(np.std(time_differences))
        final = df.loc[last_clickout_indices].drop(['timestamp'], axis=1)
        final['mean_time_per_step'] = means
        final['frenzy_factor'] = variances
        final.mean_time_per_step.fillna(final.mean_time_per_step.mean(), inplace=True)
        final.frenzy_factor.fillna(final.frenzy_factor.mean(), inplace=True)
        return final.reset_index(drop=True)


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = FrenzyFactorSession(mode=mode, cluster='no_cluster')
    c.save_feature()
