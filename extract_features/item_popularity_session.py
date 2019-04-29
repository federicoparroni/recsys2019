from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()


class ItemPopularitySession(FeatureBase):

    """
    the popularity of an impression in the context of a session
    the popularity depends on the session other than in the item because the element clicked as
    last in the session shouldnt be counted in the popularity of the same, in order to be unbiased
    during the training
    | user_id | session_id | item_id | popularity
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'item_popularity_session'
        super(ItemPopularitySession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def build_popularity(df, popularity):
            df = df[(df['action_type'] == 'clickout item')
                & (~df['reference'].isnull())]
            clicked_references = list(map(int, list(df['reference'].values)))
            frequence = list(map(int, list(df['frequence'].values)))
            for i in tqdm(range(len(clicked_references))):
                e = clicked_references[i]
                f = frequence[i]
                if int(e) in popularity:
                    popularity[int(e)] += int(f)
                else:
                    popularity[int(e)] = int(f)
            return popularity

        def func(x):
            r = []
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0].split('|')
                for i in impr:
                    if i == clk.reference.values[0]:
                        r.append((i, popularity[int(i)]-1))
                    else:
                        r.append((i, popularity[int(i)]))
            return r

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        # initialize the popularity with all the item ids: I retrieve those from accomodations_df
        s = set(data.accomodations_df().item_id.values)
        popularity = dict.fromkeys(s, 0)
        popularity = build_popularity(df, popularity)

        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars = ['user_id', 'session_id'], value_name = 'tuple').sort_values(by=['user_id', 'session_id']).dropna()
        s[['item_id', 'popularity']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        s = s.drop(['variable', 'tuple'], axis=1)
        s = s.reset_index(drop=True)
        return s


if __name__ == '__main__':
    c = ItemPopularitySession(mode='small', cluster='no_cluster')
    c.save_feature()
