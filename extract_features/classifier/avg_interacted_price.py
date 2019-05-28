import math
from numpy import mean

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class AvgInteractedPrice(FeatureBase):
    """
    | user_id | session_id | avg_interacted_price
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'avg_interacted_price'
        super(AvgInteractedPrice, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        target_indices = list(df[df.action_type == "clickout item"].drop_duplicates("session_id", keep="last").index)
        sessions = set(df.loc[target_indices, "session_id"])
        temp = df[df.session_id.isin(sessions)]
        groups = temp.groupby("session_id", sort=False)
        avg_interacted_price = list()
        for name, group in tqdm(groups, desc="Scanning sessions..."):
            price_sum = 0
            n_prices = 0
            interacted_impressions = dict()
            for t in zip(group.reference, group.impressions, group.prices):
                if type(t[0]) == str and t[0].isdigit():
                    ref = int(t[0])
                    if ref not in interacted_impressions:
                        interacted_impressions[ref] = 0
                    interacted_impressions[ref] += 1
                if type(t[1]) == str:
                    impressions = list(map(int, t[1].split("|")))
                    prices = list(map(int, t[2].split("|")))
                    for k, v in interacted_impressions.items():
                        if k in impressions:
                            n_prices += v
                            price_sum += (prices[impressions.index(k)] * v)
                    interacted_impressions = dict()
            if n_prices != 0:
                avg_interacted_price.append(price_sum / n_prices)
            else:
                avg_interacted_price.append(0)

        temp = df.loc[target_indices, ["user_id", "session_id"]]
        temp["avg_interacted_price"] = avg_interacted_price
        return temp


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = AvgInteractedPrice(mode=mode, cluster='no_cluster')
    c.save_feature()
