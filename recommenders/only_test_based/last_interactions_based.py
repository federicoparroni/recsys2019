import time

import numpy as np

# creating URM
import pandas as pd
from tqdm import tqdm
import pickle
import data

from recommenders.recommender_base import RecommenderBase


class LatestInteractionsRecommender(RecommenderBase):
    """
    Idea is to recommend all last interactions and clickouts appearing in impressions
    in order from most recently interacted to less recently

    Score is given following the scores_interactions list in the following way:
    most_recent1: 1
    most_recent2: 0.75
    most_recent3: 0.5
    ...
    """

    def __init__(self, mode, cluster='no_cluster', k_first_only_to_recommend=15):
        name = 'Last Interactions recommender'
        super(LatestInteractionsRecommender, self).__init__(mode, cluster, name)
        self.k_first_only_to_recommend = k_first_only_to_recommend

        self.weight_per_position = [53.59, 6.714, 2.38, 1.17, 0.64, 0.38, 0.22, 0.16,
                                    0.105, 0.09, 0.07, 0.6, 0.5, 0.3, 0.1]

    def _set_no_reordering(self, seq):
        """
        Remove duplicates maintaining ordering
        """
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def fit(self):

        df_test = data.test_df(self.mode, cluster=self.cluster)

        print("{}: creating grouped sessions with interaction lists".format(self.name))
        session_groups = self.get_groupby_sessions_references(data.test_df(self.mode, cluster=self.cluster))

        # Getting target sessions
        target_indices = data.target_indices(self.mode, self.cluster)

        df_test_target = df_test[df_test.index.isin(target_indices)]

        # I must reason with session_ids since i'm interested in getting last interactions of same session
        df_test_target = df_test_target.set_index("session_id")

        # then i create a dictionary for re-mapping session into indices
        if len(df_test_target.index) != len(target_indices):
            print("Indices not same lenght of sessions, go get some coffee...")
            return

        self.dictionary_indices = dict(zip(df_test_target.index, target_indices))

        list_sessions = session_groups.index

        recs_tuples = []

        print("{}: fitting the model".format(self.name))
        for i in tqdm(df_test_target.index):
            # Check if it is a session without interactions
            if i not in list_sessions:
                recs_tuples.append((self.dictionary_indices.get(i), []))
            else:
                # Get interacted element of session with no duplicates
                interacted_elements = np.asarray(session_groups.at[i, "sequence"])

                interacted_elements = np.asarray(self._set_no_reordering(x for x in interacted_elements))

                impressions = np.asarray(df_test_target.at[i, "impressions"].split("|"))

                # First i want to be sure the impressions contains all the interacted elements (if not, they must be cutted off from relevant items)
                mask_only_in_impression = np.in1d(interacted_elements, impressions, assume_unique=True)

                interacted_elements = interacted_elements[mask_only_in_impression]

                # I append the last interacted elements as first (so I invert the order of relevant_elements!)
                real_recommended = np.flipud(interacted_elements)

                real_recommended = real_recommended.astype(np.int)

                recs_tuples.append(
                    (self.dictionary_indices.get(i), list(real_recommended)[:self.k_first_only_to_recommend]))

        self.recs_batch = recs_tuples

    def recommend_batch(self):
        return self.recs_batch

    def get_scores_batch(self):
        recs_batch = self.recommend_batch()

        recs_scores_batch = []
        print("{}: getting the model scores".format(self.name))
        for tuple in recs_batch:
            recs_scores_batch.append((tuple[0], tuple[1], self.weight_per_position[:len(tuple[1])]))

        return recs_scores_batch

    def get_groupby_sessions_references(self, df_test):
        """
        Creates a groupby session_id - list of interactions ordered
        :param df_test: dataframe containing test sessions
        :return: dataframe with grouped interactions
        """

        # Leave only numeric references
        df_test = df_test[pd.to_numeric(df_test['reference'], errors='coerce').notnull()]
        df_test = df_test.drop(["action_type", "impressions"], axis=1)
        groups = df_test.groupby(['session_id'])

        aggregated = groups['reference'].agg({'sequence': lambda x: list(map(str, x))})

        return aggregated
