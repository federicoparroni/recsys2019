import numpy as np
from tqdm import tqdm
import data
from recommenders.recommender_base import RecommenderBase


class MinPriceRecommender(RecommenderBase):
    """
    It recommends all impressions for each session ordered by
    min price associated to each impression
    Score is given following the scores_impression list in the following way:
    min_price1: 1
    min_price2: 0.75
    min_price3: 0.5
    ...
    """

    def __init__(self, mode, cluster='no_cluster'):

        name = 'Ordered Impressions recommender'
        super(MinPriceRecommender, self).__init__(mode, cluster, name)

        self.mode = mode

        self.weight_per_position = [1, 0.75, 0.5, 0.33, 0.25, 0.2, 0.15, 0.125, 0.1,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def fit(self):
        """
        Create list of tuples for recommendations ordering them by impressions
        """

        df_test = data.test_df(self.mode)

        target_indices = data.target_indices(self.mode, self.cluster)

        df_test_target = df_test[df_test.index.isin(target_indices)]

        # Initializing list of recs
        recs_tuples = []

        for i in tqdm(df_test_target.index):

            impressions = df_test_target.at[i, "impressions"]
            impressions = list(map(int, impressions.split('|')))

            prices = df_test_target.at[i, "prices"]
            prices = list(map(int, prices.split('|')))

            temp_dict = {}

            for j in range(len(impressions)):
                temp_dict[impressions[j]] = int(prices[j])

            ordered_recs = sorted(temp_dict, key=temp_dict.__getitem__)

            recs_tuples.append((i, ordered_recs))

        self.recs_batch = recs_tuples

        return recs_tuples

    def recommend_batch(self):
        return self.recs_batch

    def get_scores_batch(self):
        recs_batch = self.recommend_batch()

        recs_scores_batch = []
        print("{}: getting the model scores".format(self.name))
        for tuple in recs_batch:
            recs_scores_batch.append((tuple[0], tuple[1], self.weight_per_position[:len(tuple[1])]))

        return recs_scores_batch
