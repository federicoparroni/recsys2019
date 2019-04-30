from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import os
#os.chdir("../")
#print(os.getcwd())

class AvgPriceAndPricePosition(FeatureBase):

    """
    avg position of the item clicked AND interacted by the user during the session sorted by price ascendent. -1 if no other interaction is present
    | user_id | session_id | mean_price_interacted | mean_cheap_pos_interacted
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'average_price_position_and_price_interacted_item'
        super(AvgPriceAndPricePosition, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            y = x[x['action_type'] == 'clickout item']

            impressions_prices_available = y[y['impressions'] != None][["impressions", "prices"]].drop_duplicates()
            # [13, 43, 4352, 543, 345, 3523] impressions
            # [45, 34, 54, 54, 56, 54] prices
            # -> [(13,45), (43,34), ...]
            # Then create dict
            # {13: 45, 43: 34, ... }

            tuples_impr_prices = []
            tuples_impr_price_pos_asc = []
            for i in impressions_prices_available.index:
                impr = impressions_prices_available.at[i, 'impressions'].split('|')
                prices = impressions_prices_available.at[i, 'prices'].split('|')
                tuples_impr_prices += list(zip(impr, prices))

                sorted(tuples_impr_prices, key=lambda x: x[1])
                tuples_impr_price_pos_asc += list(zip(impr, list(range(1, len(tuples_impr_prices) + 1))))

            tuples_impr_prices = list(set(tuples_impr_prices))
            dict_impr_price = dict(tuples_impr_prices)

            # Create dict for getting position wrt clicked impression based on cheapest item
            tuples_impr_price_pos_asc = list(set(tuples_impr_price_pos_asc))
            dict_impr_price_pos = dict(tuples_impr_price_pos_asc)

            sum_price = 0
            sum_pos_price = 0
            count_interacted = 0

            # IMPORTANT: I decided to consider impressions and clickouts distinctively.
            # If an impression is also clicked, that price counts double
            df_only_numeric = x[pd.to_numeric(x['reference'], errors='coerce').notnull()][
                ["reference", "impressions", "action_type"]].drop_duplicates()

            # Not considering last clickout in the train sessions
            clks_num_reference = df_only_numeric[df_only_numeric['action_type'] == 'clickout item']
            if len(y) > 0 and len(clks_num_reference) == len(y):  # is it a train session?
                idx_last_clk = y.tail(1).index.values[0]
                df_only_numeric = df_only_numeric.drop(idx_last_clk)

            for idx, row in df_only_numeric.iterrows():
                reference = row.reference
                if reference in dict_impr_price.keys():
                    if row.action_type == "clickout item":
                        sum_price += int(dict_impr_price[reference]) * 2
                        sum_pos_price += int(dict_impr_price_pos[reference]) * 2
                        count_interacted += 2
                    else:
                        sum_price += int(dict_impr_price[reference])
                        sum_pos_price += int(dict_impr_price_pos[reference])
                        count_interacted += 1

            if count_interacted > 0:
                mean_cheap_position = round(sum_pos_price / count_interacted, 2)
                mean_price_interacted = round(sum_price / count_interacted, 2)
            else:
                mean_cheap_position = -1
                mean_price_interacted = -1

            return mean_price_interacted, mean_cheap_position

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)

        return pd.DataFrame({'user_id': [x[0] for x in s.index.values], 'session_id': [x[1] for x in s.index.values],
                             'mean_price_interacted': [x[0] for x in s.values], 'mean_cheap_pos_interacted': [x[1] for x in s.values]})

if __name__ == '__main__':
    c = AvgPriceAndPricePosition(mode='full', cluster='no_cluster')
    c.save_feature()
