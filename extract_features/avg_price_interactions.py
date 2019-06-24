from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions
from extract_features.rnn.impressions_average_price import ImpressionsAveragePrice
import numpy as np

class AvgPriceInteractions(FeatureBase):

    """
        for every session, tells the avg price of the references with which the user interacted
        and their mean pos in price

        user_id | session_id | mean_cheap_pos_interacted | mean_price_interacted
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'avg_price_interactions'
        super(AvgPriceInteractions, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(self.mode, cluster=self.cluster)
        test = data.test_df(self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        idxs_click = sorted(find_last_clickout_indices(df))

        # for every last clickout index, retrieve the list 
        # of all the clickouts for that session
        list_impres = []
        list_prices_impres_wise = []
        list_prices_orderd_wise = []
        for i in tqdm(idxs_click):
            a_user = df.at[i, 'user_id']
            a_sess = df.at[i, 'session_id']
            impres = [list(map(int, df.at[i, 'impressions'].split('|')))]
            prices = list(map(int, df.at[i, 'prices'].split('|')))
            prices_impres_wise = [prices]
            prices_orderd_wise = [sorted(prices)]
            j = i-1
            while j >= 0:
                try:
                    n_user = df.at[j, 'user_id']
                    n_sess = df.at[j, 'session_id']
                    if a_sess == n_sess and a_user == n_user:
                        if df.at[j, 'action_type'] == 'clickout item':
                            impres.append(list(map(int, df.at[j, 'impressions'].split('|'))))
                            prices = list(map(int, df.at[j, 'prices'].split('|')))
                            prices_impres_wise.append(prices)
                            prices_orderd_wise.append(sorted(prices))
                    else:
                        break
                    j -= 1
                except:
                    j -= 1
            list_impres.append(impres)
            list_prices_impres_wise.append(prices_impres_wise)
            list_prices_orderd_wise.append(prices_orderd_wise)

        # then build the feature
        list_mean_prices_interacted = []
        list_mean_pos_interacted= []
        count = 0
        for i in tqdm(idxs_click):
            prices_interacted = []
            pos_interacted = []
            a_user = df.at[i, 'user_id']
            a_sess = df.at[i, 'session_id']
            impres = list_impres[count]
            prices_impres_wise = list_prices_impres_wise[count]
            prices_orderd_wise = list_prices_orderd_wise[count]
            j = i-1
            while j >= 0:
                try:
                    n_user = df.at[j, 'user_id']
                    n_sess = df.at[j, 'session_id']
                    if a_sess == n_sess and a_user == n_user:
                        n_ref = df.at[j, 'reference']
                        if n_ref.isdigit():
                            n_ref = int(n_ref)
                            count_clickouts = 0 
                            while True:
                                elem_impres = impres[count_clickouts]
                                elem_prices_impres_wise = prices_impres_wise[count_clickouts]
                                elem_prices_orderd_wise = prices_orderd_wise[count_clickouts]
                                if n_ref in elem_impres:
                                    price_reference = elem_prices_impres_wise[elem_impres.index(n_ref)]
                                    prices_interacted.append(price_reference)
                                    pos_interacted.append(elem_prices_orderd_wise.index(price_reference) + 1)
                                    break
                                else:
                                    count_clickouts += 1
                        j -= 1
                        
                    else:
                        break
                except:
                    j -= 1

            if len(prices_interacted) > 0:
                list_mean_prices_interacted.append(sum(prices_interacted)/len(prices_interacted))
            else:
                list_mean_prices_interacted.append(-1)

            if len(pos_interacted) > 0:
                list_mean_pos_interacted.append(sum(pos_interacted)/len(pos_interacted))
            else:
                list_mean_pos_interacted.append(-1)

            count += 1


        final_df = df[['user_id', 'session_id']].loc[idxs_click]
        final_df['mean_cheap_pos_interacted'] = list_mean_pos_interacted
        final_df['mean_price_interacted'] = list_mean_prices_interacted
        return final_df.reset_index(drop=True)

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection
    mode = mode_selection()
    cluster = cluster_selection()
    c = AvgPriceInteractions(mode=mode, cluster=cluster)
    c.save_feature()

