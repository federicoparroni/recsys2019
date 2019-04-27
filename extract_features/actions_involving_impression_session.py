from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()


class ActionsInvolvingImpressionSession(FeatureBase):

    """
    the kind of last actions with which the user interacted with the impression before the clickout
    | user_id | session_id | item_id | clickout item | interaction item deals | ...
    for any kind of action with potential numeric reference, the count of the times the user interacted with
    the impression in the context of that action
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'actions_involving_impression_session'
        super(ActionsInvolvingImpressionSession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0].split('|')
                r = pd.DataFrame({
                    'item_id': np.zeros(len(impr), dtype=np.int),
                    'clickout item': np.zeros(len(impr), dtype=np.int),
                    'interaction item deals': np.zeros(len(impr), dtype=np.int),
                    'interaction item image': np.zeros(len(impr), dtype=np.int),
                    'interaction item info': np.zeros(len(impr), dtype=np.int),
                    'interaction item rating': np.zeros(len(impr), dtype=np.int),
                    'search for item': np.zeros(len(impr), dtype=np.int),
                    'no action': np.zeros(len(impr), dtype=np.int)})

                x = x.loc[head_index.values[0]:clk.index.values[0]-1]
                df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][[
                    "reference", "action_type", "frequence"]]
                refs = list(df_only_numeric.reference.values)
                freqs = list(df_only_numeric.frequence.values)
                actions = list(df_only_numeric.action_type.values)
                count = 0
                for i in impr:
                    if i in refs:
                        occ = [j for j, x in enumerate(refs) if x == i]
                        for o in occ:
                            r[actions[o]][count] += freqs[o]
                    else:
                        r['no action'][count] += 1
                    r['item_id'][count] = i
                    count += 1
                return r

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(
            ['user_id', 'session_id']).progress_apply(func)
        s = s.reset_index()
        s = s.drop(['level_2'], axis=1)
        return s


if __name__ == '__main__':
    c = ActionsInvolvingImpressionSession(mode='small', cluster='no_cluster')
    c.save_feature()
