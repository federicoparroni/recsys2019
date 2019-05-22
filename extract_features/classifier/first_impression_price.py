import math

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class FirstImpressionPrice(FeatureBase):
    """
    count number of interactions with the first impression during the session
    | user_id | session_id | first_impression_price | price_position_in_ascending_order
    num_interaction_with_first_impression is int
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'first_impression_price'
        super(FirstImpressionPrice, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        target_indices = list(df[df.action_type == "clickout item"].drop_duplicates("session_id", keep="last").index)

        temp = df.loc[target_indices]
        first_prices = list()
        positions = list()
        for t in tqdm(zip(temp["session_id"], temp["prices"]), desc="Retreiving first impression's price per clickout"):
            prices = list(map(int, t[1].split("|")))
            p = prices[0]
            prices = sorted(prices)
            first_prices.append(p)
            positions.append(prices.index(p))

        if len(target_indices) != len(first_prices):
            print("Something went wrong, blame Piccio")
            exit(69)

        temp = df.loc[target_indices, ["user_id", "session_id"]]
        temp["first_price_impression"] = first_prices
        temp["price_position"] = positions
        return temp


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = FirstImpressionPrice(mode=mode, cluster='no_cluster')
    c.save_feature()
