from recommenders.recommender_base import RecommenderBase
import data
from tqdm import tqdm
import pandas as pd
import numpy as np

"""
    Boosts the recommendations gave by a recommender considering the avarage
    price of accomodations selected by the same user
"""

class BoostByPriceMean(RecommenderBase):

    def __init__(self, recomm_to_boost):
        super(BoostByPriceMean, self).__init__(mode=recomm_to_boost.mode, name='Boost_by_price_mean_of:_{}'.format(recomm_to_boost.name))
        self.recomm_to_boost = recomm_to_boost

    def fit(self):
        self.recomm_to_boost.fit()

    def recommend_batch(self):
        recs = self.recomm_to_boost.recommend_batch()
        new_recs = []

        print('rerank by price mean ..')

        # prepare df containing past clickouts from train and test but with reference
        df = data.train_df(self.mode)
        df = df[df['action_type'] == 'clickout item']
        df = df[~df['reference'].isnull()]
        df_test = data.test_df(self.mode)
        df_test = df_test[df_test['action_type'] == 'clickout item']
        df_test = df_test[~df_test['reference'].isnull()]
        df = pd.concat([df, df_test], ignore_index=True)

        # optimization: remove all the users not presents on the handle,
        # i.e. the ones that do not have a missing reference of clickout on test
        df_handle = data.handle_df(self.mode)
        df = df[df['user_id'].isin(df_handle['user_id'].values)]

        df_test = data.test_df(self.mode)
        df_test = df_test[df_test['action_type'] == 'clickout item']
        df_test = df_test[df_test['reference'].isnull()]

        # optimization: lets create a new df_test for direct indexing
        df_test_access = df_test.set_index('session_id')

        # data says that with this value of diff between a price and the user_mean_price
        # there is 0.85 probability that the user wont pick that appartament
        confidence_std = 20

        for r in tqdm(recs):
            session_id = r[0]

            user_id = df_test_access.loc[session_id]['user_id']
            past_clickouts = df[np.in1d(df['user_id'].values, [user_id])]

            acc = 0
            iters = 0
            for idx, row in past_clickouts.iterrows():
                impression = list(map(int, row['impressions'].split('|')))
                prices = list(map(int, row['prices'].split('|')))
                clickout_item = int(row['reference'])
                if clickout_item in impression:
                    price = prices[impression.index(clickout_item)]
                    acc += price
                iters += 1

            if iters > 0:
                user_mean_price = acc/iters
                new_rank = []
                row = df_test[df_test['session_id'] == session_id]
                impression = list(map(int, row['impressions'].values[0].split('|')))
                prices = list(map(int, row['prices'].values[0].split('|')))
                j = 0
                for i in range(len(r[1]) - 1, -1, -1):
                    price = prices[impression.index(r[1][i])]
                    if abs(price - user_mean_price) >= confidence_std:
                        new_rank.append(r[1][i])
                    else:
                        new_rank.insert(j, r[1][i])
                        j += 1
                
                new_recs.append((session_id, new_rank))
            else:
                new_recs.append(r)

        return new_recs
