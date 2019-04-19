from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class TimesUserInteractedWithImpression(FeatureBase):

    """
    for any unique session, tells how many times the user interacted with any single impression
    among the ones that compare on the clickout
    | user_id | session_id | item_id | n_times_clicked_before_clk
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'times_user_interacted_with_impression'
        super(TimesUserInteractedWithImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            r = []
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0].split('|')
                print(f'x before {x}')
                x = x.loc[head_index.values[0]:clk.index.values[0]-1]
                print(f'x after {x}')
                df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][["reference", "action_type", "frequence"]]
                refs = list(df_only_numeric.reference.values)
                freqs = list(df_only_numeric.frequence.values)
                for i in impr:
                    if i in refs:
                        occ = [j for j, x in enumerate(refs) if x == i]
                        count = 0
                        for o in occ:
                            count += freqs[o]
                        r.append((i, count))
                    else:
                        r.append((i, 0))
            return r

        def count_freq(x):
            """
            Versione fixata di quella sopra che conta le occorrenze per ogni impression
            negli step precedenti della sessione.
            TODO: se va bene questa cancellate quella sopra
            """
            r = []
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                x = x[x['step']<int(clk['step'])]
                df_only_numeric = x[x['reference'].astype(str).str.isdigit()]
                refs = []
                if df_only_numeric.shape[0]>0:
                    refs = list(df_only_numeric.reference.values)
                impr = clk.impressions.values[0].split('|')
                for i in impr:
                    if i in refs:
                        occ = refs.count(i)
                        r.append((i, occ))
                    else:
                        r.append((i, 0))
            return r

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(count_freq)
        s = s.apply(pd.Series).reset_index().melt(id_vars = ['user_id', 'session_id'], value_name = 'tuple').sort_values(by=['user_id', 'session_id']).dropna()
        s[['item_id', 'n_times_clicked_before_clk']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        s = s.drop(['variable', 'tuple'], axis=1)
        s = s.reset_index(drop=True)
        return s

if __name__ == '__main__':
    c = TimesUserInteractedWithImpression(mode='small', cluster='no_cluster')
    c.save_feature()
