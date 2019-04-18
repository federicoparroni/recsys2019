from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class SessionActionNumRefDiffFromImpressions(FeatureBase):

    """
    actions that happened:
    | user_id | session_id | session_action_num_ref_diff_from_impressions
    session_action_num_ref_diff_from_impressions is string, in the format: action1 | action2 | action3 ...
    it is possible to have more times a action as replicated
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'session_action_num_ref_diff_from_impressions'
        columns_to_onehot = [('session_action_num_ref_diff_from_impressions', 'multiple')]
        super(SessionActionNumRefDiffFromImpressions, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):

        def func(x):
            s = ''
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0]
                x = x.loc[head_index.values[0]:clk.index.values[0]-1]
                df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][["reference", "action_type"]]
                df_only_numeric = df_only_numeric.reset_index(drop=True)
                refs = df_only_numeric.reference.values
                for i in range(len(refs)):
                    if refs[i] not in impr:
                        s += df_only_numeric.loc[i].action_type + '|'
            if s == '':
                return 'no_action'
            else:
                return s[:-1]

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        return pd.DataFrame({'user_id':[x[0] for x in s.index.values], 'session_id':[x[1] for x in s.index.values], 'session_action_num_ref_diff_from_impressions': s.values})

if __name__ == '__main__':
    c = SessionActionNumRefDiffFromImpressions(mode='small', cluster='no_cluster')
    c.save_feature()
