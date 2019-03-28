import time

import numpy as np

# creating URM
import pandas as pd
from tqdm import tqdm
import pickle
import data

from recommenders.recommender_base import RecommenderBase

scores_interactions = [1, 0.5, 0.33, 0.25, 0.2,  0.166, 0.15, 0.125,
                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

class LatestInteractionsRecommender(RecommenderBase):
    """
    Idea is to recommend all last interactions and clickouts appearing in impressions
    in order from most recently interacted to less recently
    """
    def __init__(self, mode):
        name = 'Last Interactions recommender'
        super(LatestInteractionsRecommender, self).__init__(mode, name)

        self.mode = mode


    def f7(self, seq):
        """
        Remove duplicates maintaining ordering
        """
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def fit(self):

        df_test = data.test_df(self.mode)

        print("{}: creating grouped sessions with interaction lists".format(self.name))
        session_groups = self.get_groupby_sessions_references(data.test_df(self.mode))

        # Getting target sessions
        df_test_target = df_test[df_test["action_type"] == "clickout item"]

        df_test_target = df_test_target.replace(r'', np.nan, regex=True)

        df_test_target = df_test_target[~df_test_target.reference.notnull()]

        df_test_target = df_test_target.set_index("session_id")

        list_sessions = session_groups.index

        recs_tuples = []
        print("{}: fitting the model".format(self.name))
        for i in tqdm(df_test_target.index):
            if i not in list_sessions:
                recs_tuples.append((i, []))
            else:
                # Get interacted element of session with no duplicates
                interacted_elements = np.asarray(session_groups.at[i, "sequence"])

                interacted_elements = np.asarray(self.f7(x for x in interacted_elements))

                # I append the last interacted elements as first (so I invert the order of relevant_elements!)
                real_recommended = np.flipud(interacted_elements)

                recs_tuples.append((i, real_recommended))

        self.recs_batch = recs_tuples


    def recommend_batch(self):
        return self.recs_batch

    def get_scores_batch(self):
        recs_batch = self.recommend_batch()

        recs_scores_batch = []
        print("{}: getting the model scores".format(self.name))
        for tuple in recs_batch:
            recs_scores_batch.append((tuple[0], tuple[1], scores_interactions[:len(tuple[1])]))

        return recs_scores_batch


    def get_groupby_sessions_references(self, df_test):
        """
        Creates a groupby session_id - list of interactions ordered
        :param df_test: dataframe containing test sessions
        :return: dataframe with grouped interactions
        """

        #Leave only numeric references
        df_test = df_test[pd.to_numeric(df_test['reference'], errors='coerce').notnull()]
        df_test = df_test.drop(["action_type", "impressions"], axis=1)
        groups = df_test.groupby(['session_id'])

        aggregated = groups['reference'].agg({'sequence': lambda x: list(map(str, x))})

        return aggregated