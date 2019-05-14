import numpy as np
import pandas as pd
from tqdm import tqdm
import math

import data

from recommenders.recommender_base import RecommenderBase

class LazyUserRecommender(RecommenderBase):
    '''
        This is a bet on the user's lazyness.
        Key idea:   assuming as the N-th action the clickout to be predicted, looking at the N-1-th interaction it's
                    possible to infer in which position, inside the page, the user is. Then, recommend only items which
                    are in a position greater or equal than the inferred position.
        Score:      public 0.64
                    small 0.638
                    local 0.643
    '''


    def __init__(self, mode, cluster='no_cluster', time_delay_treshold = 5000, weight_per_position = [1] * 25):
        name = 'Lazy User Recommender'
        super(LazyUserRecommender, self).__init__(mode, cluster, name)
        self.time_delay_treshold = time_delay_treshold
        self.weight_per_position = [53.59, 9.714, 4.38, 2.17, 1.64,  0.58, 0.22, 0.16,
                       0.105, 0.09, 0.07, 0.6, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.recs = None

    def fit(self):
        df_test = data.test_df(self.mode, cluster=self.cluster)

        # Getting target sessions
        target_indices = data.target_indices(self.mode, self.cluster)

        df_test_target = df_test[df_test.index.isin(target_indices)]
        count = 0
        oneshot = 0
        resorted = 0
        recs_tuples = []
        print("Fitting...")
        for index, row in tqdm(df_test_target.iterrows()):
            impressions = list(map(int, row["impressions"].split("|")))
            if int(row.step) == 1:
                # This means that the clickout is the first interaction of the session --> we are in a one shot session
                oneshot += 1
                recs_tuples.append((index, impressions))
            else:
                previous_row = df_test.loc[index - 1]
                last_interaction_ref = previous_row["reference"]
                if type(last_interaction_ref) == str and last_interaction_ref.isdigit():
                    last_interaction_ref = int(last_interaction_ref)
                if last_interaction_ref and last_interaction_ref in impressions:
                    i = impressions.index(last_interaction_ref)
                    sorted_impressions = impressions[i:] + impressions[:i]
                    count += 1
                    recs_tuples.append((index, sorted_impressions))
                else:
                    resorted += 1
                    recs_tuples.append((index, impressions))

        print("{} % of resorted session".format(round(resorted / len(df_test_target) * 100, 2)))
        print("{} % of oneshot session".format(round(oneshot / len(df_test_target) * 100, 2)))
        print("{} % of lazy session".format(round(count / len(df_test_target) * 100, 2)))
        self.recs = recs_tuples
        print("Fitting completed!")

    def recommend_batch(self):
        if self.recs is None:
            self.fit()
        return self.recs

    def get_scores_batch(self):
        if self.recs is None:
            self.fit()
        recs_batch = self.recs

        recs_scores_batch = []
        print("{}: getting the model scores".format(self.name))
        for rec in recs_batch:
            recs_scores_batch.append((rec[0], rec[1], self.weight_per_position[:len(rec[1])]))

        return recs_scores_batch


