from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class PricePositionInfoInteractedReferences(FeatureBase):

    """
    Avg position of the item clicked AND interacted by the user with respect to the price position sorted by the cheapest
    during the session sorted by price ascendent.
    -1 if no other interaction is present.
    Position of the impressions interacted usually when info about impression is available. (number from 1 to 25)
    also position of the last impression interacted/clicked (this hopes to let apply what lazy user recommender does)
    -1 is not available.
    | user_id | session_id | mean_price_interacted | mean_cheap_pos_interacted | 'mean_pos' | 'pos_last_reference'
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'price_position_info_interacted_references'
        super(PricePositionInfoInteractedReferences, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def func(x):
            y = x[x['action_type'] == 'clickout item']

            mean_pos = -1
            pos_last_reference = -1
            mean_cheap_position = -1
            mean_price_interacted = -1

            if len(y) > 0:
                clk = y.tail(1)
                head_index = x.head(1).index
                x = x.loc[head_index.values[0]:clk.index.values[0] - 1]

                if len(x) > 1:
                    impressions_pos_available = y[['impressions', 'prices']].drop_duplicates()

                    # [13, 43, 4352, 543, 345, 3523] impressions
                    # [45, 34, 54, 54, 56, 54] prices
                    # -> [(13,45), (43,34), ...]
                    # Then create dict
                    # {13: 45, 43: 34, ... }

                    tuples_impr_prices = []
                    tuples_impr_price_pos_asc = []

                    # [13, 43, 4352, 543, 345, 3523] impressions
                    # Then create dict impression-position
                    # {13: 1, 43: 2, ... }
                    tuples_impr_pos = []

                    for i in impressions_pos_available.index:
                        impr = impressions_pos_available.at[i, 'impressions'].split('|')
                        prices = list(map(int,impressions_pos_available.at[i, 'prices'].split('|')))
                        tuples_impr_prices += list(zip(impr, prices))

                        tuples_impr_pos += [(impr[idx], idx + 1) for idx in range(len(impr))]

                        prices_sorted = prices.copy()
                        prices_sorted.sort()

                        tuples_impr_price_pos_asc += [(impr[idx], prices_sorted.index(prices[idx]) + 1) for idx in range(len(impr))]

                    # dictionary: from impression, get its price
                    dict_impr_price = dict(list(set(tuples_impr_prices)))

                    # dictionary: from impression, get its position on impression
                    dict_impr_pos = dict(list(set(tuples_impr_pos)))

                    # dictionary: from impression, get its price position wrt the ascending price order
                    dict_impr_price_pos = dict(list(set(tuples_impr_price_pos_asc)))

                    # IMPORTANT: I decided to consider impressions and clickouts distinctively.
                    # If an impression is also clicked, that price counts double
                    # considering reference, impressions and action type as a row, I can distinguish from clickouts and impressions dropping duplicates
                    df_only_numeric = x[["reference", "impressions", "action_type"]].drop_duplicates()


                    sum_price = 0
                    sum_pos_price = 0
                    sum_pos_impr = 0
                    count_interacted_pos_impr = 0
                    count_interacted = 0
                    for i in df_only_numeric.index:
                        reference = df_only_numeric.at[i, 'reference']

                        if reference in dict_impr_price.keys():
                            sum_pos_impr += int(dict_impr_pos[reference])
                            sum_price += int(dict_impr_price[reference])
                            sum_pos_price += int(dict_impr_price_pos[reference])
                            count_interacted_pos_impr += 1
                            count_interacted += 1

                    if count_interacted > 0:
                        mean_cheap_position = round(sum_pos_price / count_interacted, 2)
                        mean_price_interacted = round(sum_price / count_interacted, 2)
                        mean_pos = round(sum_pos_impr / count_interacted_pos_impr, 2)

                        last_reference = df_only_numeric.tail(1).reference.values[0]
                        # Saving the impressions appearing in the last clickout (they will be used to get the 'pos_last_reference'
                        impressions_last_clickout = clk.impressions.values[0].split('|')
                        if last_reference in impressions_last_clickout:
                            pos_last_reference = impressions_last_clickout.index(last_reference) + 1

            return pd.Series({'mean_cheap_position': mean_cheap_position, 'mean_price_interacted':mean_price_interacted,
                             'mean_pos': mean_pos, 'pos_last_reference': pos_last_reference})

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        s = df.groupby(['user_id', 'session_id']).progress_apply(func)

        return s.reset_index()
if __name__ == '__main__':
    c = PricePositionInfoInteractedReferences(mode='small', cluster='no_cluster')
    c.save_feature()
