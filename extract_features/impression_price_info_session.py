from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import os
#os.chdir("../")
#print(os.getcwd())

class ImpressionPriceInfoSession(FeatureBase):

    """
    absolute price of impression as it appears in the last clickut of a session and
    the position in price
    | user_id | session_id | item_id | impression_price | impression_position_in_price
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_price_info_session'
        super(ImpressionPriceInfoSession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            r = []
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                impr = clk.impressions.values[0].split('|')
                pric = list(map(int, clk.prices.values[0].split('|')))
                sorted_pric = sorted(pric)
                count = 0
                for i in impr:
                    r.append((i, pric[count], sorted_pric.index(pric[count]) + 1))
                    count += 1
            return r

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars = ['user_id', 'session_id'], value_name = 'tuple').sort_values(by=['user_id', 'session_id']).dropna()
        s[['item_id', 'impression_price', 'impression_position_in_price']] = pd.DataFrame(s['tuple'].tolist(), index=s.index)
        s = s.drop(['variable', 'tuple'], axis=1)
        s = s.reset_index(drop=True)
        return s

if __name__ == '__main__':
    c = ImpressionPriceInfoSession(mode='small', cluster='no_cluster')
    c.save_feature()
