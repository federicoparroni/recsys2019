import math
from numpy import mean

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class PriceStats(FeatureBase):
    """
    count number of interactions with the first impression during the session
    | user_id | session_id | first_impression_price | price_position_in_ascending_order
    num_interaction_with_first_impression is int
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'price_stats'
        super(PriceStats, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        target_indices = list(df[df.action_type == "clickout item"].drop_duplicates("session_id", keep="last").index)

        temp = df.loc[target_indices]
        avg_prices = list()
        min_prices = list()
        max_prices = list()
        for t in tqdm(zip(temp["session_id"], temp["prices"]), desc="Retreiving first impression's price per clickout"):
            prices = list(map(int, t[1].split("|")))
            prices = prices[1:]
            if len(prices) > 0:
                avg_prices.append(mean(prices))
                min_prices.append(min(prices))
                max_prices.append(max(prices))
            else:
                avg_prices.append(0)
                min_prices.append(0)
                max_prices.append(0)

        if len(target_indices) != len(avg_prices):
            print("Something went wrong, blame Piccio")
            exit(69)

        temp = df.loc[target_indices, ["user_id", "session_id"]]
        temp["avg_prices_other_impressions"] = avg_prices
        temp["min_price"] = min_prices
        temp["max_price"] = max_prices
        return temp


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = PriceStats(mode=mode, cluster='no_cluster')
    c.save_feature()
