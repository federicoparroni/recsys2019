from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions
from extract_features.rnn.impressions_average_price import ImpressionsAveragePrice
import numpy as np


class PriceInfoSession(FeatureBase):

    """
    | user_id | session_id |min_pos_interacted|max_pos_interacted|first_pos_interacted|last_pos_interacted
    |num_interacted_impressions|percentage_interacted_impressions
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'price_info_session'
        super(PriceInfoSession, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=[('user_class', 'single'),
                                                                      ('user_click_class', 'single')])

    def extract_feature(self):

        price_dict_df = ImpressionsAveragePrice().read_feature().set_index('item_id')
        price_dict = price_dict_df.to_dict('index')

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        temp = df.fillna('0')
        idxs_click = sorted(find_last_clickout_indices(temp))
        idxs_numeric_reference = temp[temp['reference'].str.isnumeric() == True].index

        count = 0
        last_click = idxs_click[0]

        features = []

        prices_interacted = []
        impression_interacted = {}
        prices_clickout_interacted = []

        for i in tqdm(sorted(idxs_numeric_reference)):

            if i == last_click:

                prices_click = sorted(list(map(int, temp.at[i, 'prices'].split('|'))))

                mean_price_click = np.mean(np.array(prices_click))

                max_price_click = prices_click[-1]
                min_price_click = prices_click[0]
                var_prices_click = np.var(np.array(prices_click))

                support_interaction = len(prices_interacted)

                if support_interaction == 0:
                    last_price_interacted = -1
                    mean_price_interacted = -1
                    min_price_interacted = -1
                    max_price_interacted = -1
                    user_class = 'None'
                    var_price_interacted = -1
                    distance_max_price_from_mean = -1
                else:
                    last_price_interacted = prices_interacted[-1]
                    prices_interacted = sorted(prices_interacted)
                    mean_price_interacted = np.mean(np.array(prices_interacted))
                    min_price_interacted = prices_interacted[0]
                    max_price_interacted = prices_interacted[-1]
                    user_class = 'poor' if mean_price_interacted < mean_price_click else 'rich'
                    var_price_interacted = np.var(np.array(prices_interacted))
                    distance_max_price_from_mean = max_price_interacted - mean_price_click

                support_interaction_clickout = len(prices_clickout_interacted)

                if support_interaction_clickout == 0:
                    last_price_clickout_interacted = -1
                    prices_clickout_interacted = -1
                    mean_price_clickout_interacted = -1
                    min_price_clickout_interacted = -1
                    max_price_clickout_interacted = -1
                    user_click_class = 'None'
                    var_prices_click_interacted = -1
                    distance_max_price_clickout_from_mean = -1
                else:
                    last_price_clickout_interacted = prices_clickout_interacted[-1]
                    prices_clickout_interacted = sorted(prices_clickout_interacted)
                    mean_price_clickout_interacted = np.mean(np.array(prices_clickout_interacted))
                    min_price_clickout_interacted = prices_clickout_interacted[0]
                    max_price_clickout_interacted = prices_clickout_interacted[-1]
                    user_click_class = 'poor' if mean_price_clickout_interacted < mean_price_click else 'rich'
                    var_prices_click_interacted = np.var(np.array(prices_clickout_interacted))
                    distance_max_price_clickout_from_mean = max_price_clickout_interacted - mean_price_click

                features_dict = {
                    'max_price_click': max_price_click,
                    'min_price_click': min_price_click,
                    'var_prices_click': var_prices_click,
                    'support_interaction': support_interaction,
                    'last_price_interacted': last_price_interacted,
                    'mean_price_interacted': mean_price_interacted,
                    'min_price_interacted': min_price_interacted,
                    'max_price_interacted': max_price_interacted,
                    'user_class': user_class,
                    'var_price_interacted': var_price_interacted,
                    'distance_max_price_from_mean': distance_max_price_from_mean,
                    'support_interaction_clickout': support_interaction_clickout,
                    'last_price_clickout_interacted': last_price_clickout_interacted,
                    'mean_price_clickout_interacted': mean_price_clickout_interacted,
                    'min_price_clickout_interacted': min_price_clickout_interacted,
                    'max_price_clickout_interacted': max_price_clickout_interacted,
                    'user_click_class': user_click_class,
                    'var_prices_click_interacted': var_prices_click_interacted,
                    'distance_max_price_clickout_from_mean': distance_max_price_clickout_from_mean
                }

                features.append(features_dict)

                count += 1
                prices_interacted = []
                impression_interacted = {}
                prices_clickout_interacted = []

                if count < len(idxs_click):
                    last_click = idxs_click[count]
                continue

            ref = int(temp.at[i, 'reference'])

            action_type = temp.at[i, 'action_type']
            if action_type == 'clickout item':
                prices = list(map(int, temp.at[i, 'prices'].split('|')))
                impressions = list(map(int, temp.at[i, 'impressions'].split('|')))
                idx = impressions.index(ref)
                prices_clickout_interacted.append(prices[idx])
                if ref not in impression_interacted:
                    impression_interacted[ref] = 1
                    prices_interacted.append(prices[idx])
            else:
                if ref not in impression_interacted:
                    impression_interacted[ref] = 1
                    if ref in price_dict:
                        prices_interacted.append(price_dict[ref]['prices_mean'])
                    else:
                        pass

        final_df = temp[['user_id', 'session_id']].loc[idxs_click]
        final_df['dict'] = features

        features_df = pd.DataFrame(final_df.progress_apply(lambda x: tuple(x['dict'].values()), axis=1).tolist(),
                                   columns=list(final_df.iloc[0].dict.keys()))
        final_df_ = pd.merge(final_df.drop('dict', axis=1).reset_index(drop=True).reset_index(),
                             features_df.reset_index()).drop('index', axis=1)
        return final_df_


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = PriceInfoSession(mode=mode, cluster=cluster)
    c.save_feature()

