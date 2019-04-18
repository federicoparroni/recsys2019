from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class ActionsInvolvingImpressionSession(FeatureBase):

    """
    the kind of last actions with which the user interacted with the impression before the clickout
    | user_id | session_id | item_id | actions_involving_impression_session
    actions_involving_impression_session is a string. contains pipe-separated actions and
    an action can appear multiple times
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'actions_involving_impression_session'
        columns_to_onehot = [('actions_involving_impression_session', 'multiple')]
        super(ActionsInvolvingImpressionSession, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):

        def func(x):
            r = []
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0].split('|')
                x = x.loc[head_index.values[0]:clk.index.values[0]-1]
                df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][["reference", "action_type", "frequence"]]
                refs = list(df_only_numeric.reference.values)
                freqs = list(df_only_numeric.frequence.values)
                actions = list(df_only_numeric.action_type.values)
                for i in impr:
                    if i in refs:
                        occ = [j for j, x in enumerate(refs) if x == i]
                        count = ''
                        for o in occ:
                            count += (freqs[o] * (actions[o] + '|'))
                        count = count[:-1]
                        r.append((i, count))
                    else:
                        r.append((i, 'no_action'))
            return r

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars = ['user_id', 'session_id'], value_name = 'tuple').sort_values(by=['user_id', 'session_id']).dropna()
        s[['item_id', 'actions_involving_impression_session']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        s = s.drop(['variable', 'tuple'], axis=1)
        s = s.reset_index(drop=True)
        return s
    
if __name__ == '__main__':
    c = ActionsInvolvingImpressionSession(mode='small', cluster='no_cluster')
    c.read_feature(one_hot=True)
