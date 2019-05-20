import numpy as np
from tqdm import tqdm
import data
from recommenders.recommender_base import RecommenderBase


class OrderedImpressionRecommender(RecommenderBase):
    """
    It recommends all impressions for each session ordered by
    how they are shown in Trivago during the clickout
    """

    def __init__(self, mode, cluster='no_cluster'):

        name = 'Ordered Impressions recommender'
        super(OrderedImpressionRecommender, self).__init__(mode, cluster, name)

        self.mode = mode
        self.recs_batch = None
        self.weight_per_position = [32.24, 10.63, 7.53, 6.01, 5.1, 4.13, 3.63, 3.17,
                                    2.86, 2.597, 2.33, 2.148, 1.96, 1.8, 1.669, 1.58,
                                    1.43, 1.32, 1.25, 1.166, 1.115, 1.055, 1.019, 1.007, 1.14]

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
            recs_tuples.append((i, impressions))

        self.recs_batch = recs_tuples

    def recommend_batch(self):
        if self.recs_batch is None:
            self.fit()
        return self.recs_batch

    def get_scores_batch(self):
        recs_batch = self.recommend_batch()

        recs_scores_batch = []
        print("{}: getting the model scores".format(self.name))
        for tuple in recs_batch:
            recs_scores_batch.append((tuple[0], tuple[1], self.weight_per_position[:len(tuple[1])]))

        return recs_scores_batch
