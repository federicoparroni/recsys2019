import time

import numpy as np

# creating URM
import pandas as pd
from tqdm import tqdm
import pickle
import data

from recommenders.recommender_base import RecommenderBase


class OrderedConsequentClickoutRecommender(RecommenderBase):
    """
    It recommends, for the sessions that have a clickout or interaction reference right before a missing
    clickout, the last numeric refenrences from that clickout and in order of appearence.

    Score is given following the scores_interactions list in the following way:
    most_recent1: 1
    most_recent2: 0.75
    most_recent3: 0.5
    ...
    """

    def __init__(self, mode, cluster='no_cluster', filter_only_last_clickout=True):
        name = 'Ordered consequent clickout recommender'
        super(OrderedConsequentClickoutRecommender, self).__init__(mode, cluster, name)
        self.filter_only_last_clickout = filter_only_last_clickout

        self.weight_per_position = [1, 0.75, 0.5, 0.33, 0.25, 0.2, 0.15, 0.125,
                                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]

    """ Reorder the array of elements starting from the last_elem 
        e.g.
        order_by_last_interaction([10,32,22,12], 22)
        -> [22,12,10,32]

        returns None if element is not found
    """

    def order_by_last_interaction(self, impressions, last_el):
        if last_el not in impressions:
            return impressions
        index = impressions.index(last_el) + 1

        for _ in range(index):
            impressions.append(impressions.pop(0))

        return impressions

    def fit(self):

        df_test = data.test_df(self.mode, cluster=self.cluster)

        print("{}: creating grouped sessions with interaction lists".format(self.name))

        test_co_references = df_test[pd.to_numeric(df_test['reference'], errors='coerce').notnull()]

        # se ho reference come clickout-> scrivo come reference il numero cominciando con zero
        test_co_references = test_co_references.drop(
            ["step", "timestamp", "platform", "city", "device",
             "current_filters", "prices"], axis=1)
        test_co_references = test_co_references.drop_duplicates()

        session_groups = self.get_groupby_sessions_references(test_co_references)

        if self.filter_only_last_clickout:
            # Select only sessions with final clickout
            session_groups = session_groups[session_groups["sequence"].apply(lambda x: x[-1][-1] == "o")]

        print("{}: number of sessions with final clickout right before missing reference is {}".format(self.name, len(
            session_groups)))

        # Getting target sessions
        target_indices = data.target_indices(self.mode, self.cluster)
        list_sessions = session_groups.index
        df_test_target = df_test

        self.dictionary_indices = dict(zip(df_test_target.index, df_test_target.session_id))

        recs_tuples = []
        considered = 0

        prev_impr = ""
        prev_tm = 0

        target_indices_copy = target_indices.copy()
        target_indices_copy = np.append(target_indices_copy, 0)
        print("{}: fitting the model".format(self.name))
        for i in tqdm(df_test.index):

            if i + 1 == target_indices_copy[0]:
                curr_impr = df_test_target.at[i, "impressions"]
                curr_tm = df_test_target.at[i, "timestamp"]
                prev_impr = curr_impr
                prev_tm = curr_tm

                if i == target_indices_copy[0]:
                    recs_tuples.append((i, []))
                    target_indices_copy = np.delete(target_indices_copy, np.argwhere(target_indices_copy == i))
            else:
                if i == target_indices_copy[0]:
                    target_indices_copy = np.delete(target_indices_copy, np.argwhere(target_indices_copy == i))
                    curr_impr = df_test_target.at[i, "impressions"]
                    curr_tm = df_test_target.at[i, "timestamp"]

                    if (self.dictionary_indices.get(i) not in list_sessions) or (prev_impr != curr_impr) or \
                            (prev_tm < curr_tm - 400):

                        recs_tuples.append((i, []))
                    else:
                        # Get last clickout element of session without "co"
                        last_interacted = session_groups.at[self.dictionary_indices.get(i), "sequence"][-1][:-2]

                        impressions = df_test_target.at[i, "impressions"].split("|")

                        real_recommended = np.asarray(self.order_by_last_interaction(impressions, last_interacted))

                        # Reorder impressions from last_interacted and so on:
                        real_recommended = real_recommended.astype(np.int)

                        considered += 1
                        recs_tuples.append((i, list(real_recommended)))

        print("Considered sessions are {}".format(considered))
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

        df_test.loc[df_test.action_type == 'clickout item', 'reference'] = df_test.loc[
                                                                               df_test.action_type == 'clickout item', 'reference'] + "co"

        df_test = df_test.drop(["action_type", "impressions"], axis=1)
        groups = df_test.groupby(['session_id'])

        aggregated = groups['reference'].agg({'sequence': lambda x: list(map(str, x))})

        return aggregated
