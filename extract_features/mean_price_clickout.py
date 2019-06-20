from extract_features.feature_base import FeatureBase
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
import data
import pandas as pd
import numpy as np
from tqdm import tqdm

class MeanPriceClickout(FeatureBase):
    """
    mean price among the impressions at the moment of the clickout
    | user_id | session_id | mean_price_clickout
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'mean_price_clickout'
        super(MeanPriceClickout, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        tr = data.train_df(mode=self.mode, cluster=self.cluster)
        te = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([tr, te])
        idxs = sorted(find(df))
        mean_prices = []
        for i in tqdm(idxs):
            prices = list(map(int, df.at[i, 'prices'].split('|')))
            mean_prices.append(sum(prices)/len(prices))

        total = df.loc[idxs, ['user_id', 'session_id']]
        total['mean_price_clickout'] = mean_prices
        return total.reset_index(drop=True)


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = MeanPriceClickout(mode=mode, cluster=cluster)
    c.save_feature()
