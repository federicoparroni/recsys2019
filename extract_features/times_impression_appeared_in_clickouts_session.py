from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class TimesImpressionAppearedInClickoutsSession(FeatureBase):

    """
    for any unique session, tells how many times any impression was shown to the user
    during the session ie how many times the impression was on the impressions
    of some clickouts
    | user_id | session_id | item_id | times_impression_appeared_in_clickouts_session
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'times_impression_appeared_in_clickouts_session'
        super(TimesImpressionAppearedInClickoutsSession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def count_freq(x):
            r = []
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                x = x[x['step']<int(clk['step'])]
                list_impressions = list(x[~x.impressions.isnull()].impressions.values)
                impressions = ('|'.join(list_impressions)).split('|')
                impr = clk.impressions.values[0].split('|')
                for i in impr:
                    r.append((i, impressions.count(i) + 1))
            return r

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        df = df.drop(['timestamp', 'reference', 'platform', 'city', 'device', 'current_filters', 'prices'], axis=1)
        s = df.groupby(['user_id', 'session_id']).progress_apply(count_freq)
        s = s.apply(pd.Series).reset_index().melt(id_vars = ['user_id', 'session_id'], value_name = 'tuple').sort_values(by=['user_id', 'session_id']).dropna()
        s[['item_id', 'times_impression_appeared_in_clickouts_session']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        s = s.drop(['variable', 'tuple'], axis=1)
        s = s.reset_index(drop=True)
        return s

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = TimesImpressionAppearedInClickoutsSession(mode=mode, cluster=cluster)
    c.save_feature()
