from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

tqdm.pandas()


class MeanPriceClickout(FeatureBase):
    """
    mean price of the item clicked by the user during the session, if there aren't other clickout it is equal -1
    | user_id | session_id | mean_price
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'mean price clickout'
        super(MeanPriceClickout, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            y = x[x['action_type'] == 'clickout item']
            if len(y) > 0:
                mean_price = 0
                clk = y.tail(1)
                head_index = x.head(1).index

                df = y.loc[head_index.values[0]:clk.index.values[0] - 1]
                if len(df) > 0:
                    clickouts = df[df['action_type'] == 'clickout item']
                    if len(clickouts) > 0:
                        mean_price = 0
                        for i, row in clickouts.iterrows():
                            impr = list(map(int, row.impressions.split('|')))
                            prices = list(map(int, row.prices.split('|')))
                            ref = int(row.reference)
                            if ref in impr:
                                pos = impr.index(ref)
                                mean_price += prices[pos]*row['frequence']
                        mean_price /= np.sum(np.array(clickouts['frequence']))
                if mean_price == 0:
                    mean_price = -1
                return mean_price

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)
        s = s.apply(pd.Series).reset_index().melt(id_vars=['user_id', 'session_id'], value_name='mean price') \
            .sort_values(by=['user_id', 'session_id']).dropna()
        s = s.drop('variable', axis=1)
        # create dataframe with : user_id, session_id, item_id, label (1 if it's the cliked impression, 0 otherwise)
        return s


if __name__ == '__main__':
    c = MeanPriceClickout(mode='small', cluster='no_cluster')
    c.save_feature()