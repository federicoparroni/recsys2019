import time

import numpy as np

# creating URM
import pandas as pd
from tqdm import tqdm
import pickle
import data

from recommenders.recommender_base import RecommenderBase

class LastInteractionThenAppearanceOrder(RecommenderBase):

    """
    It recommends, for the sessions that have a numeric reference before a missing
    clickout, the last numeric refenrences in order of appearence. 
    Then, it appends back the impressions in show up order

    MRR in numeric_reference_one_step_before_missing_clk: 0.67
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'last_interaction_then_appaerance_order'
        super(LastInteractionThenAppearanceOrder, self).__init__(mode, cluster, name)

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

        #I must reason with session_ids since i'm interested in getting last interactions of same session
        df_test_target = df_test_target.set_index("session_id")

        #then i create a dictionary for re-mapping session into indices
        if len(df_test_target.index) != len(target_indices):
            print("Indices not same lenght of sessions, go get some coffee...")
            return

        self.dictionary_indices = dict(zip(df_test_target.index, target_indices))

        list_sessions = session_groups.index

        recs_tuples = []
        print("{}: fitting the model".format(self.name))
        for i in tqdm(df_test_target.index):
            if i not in list_sessions:
                recs_tuples.append((self.dictionary_indices.get(i), []))
            else:
                # Get interacted element of session with no duplicates
                interacted_elements = np.asarray(session_groups.at[i, "sequence"])

                interacted_elements = np.asarray(self._set_no_reordering(x for x in interacted_elements))

                impressions = np.asarray(df_test_target.at[i, "impressions"].split("|"))
                # First i want to be sure the impressions contains all the interacted elements (if not, they must be cutted off from relevant items)
                mask_only_in_impression = np.in1d(interacted_elements, impressions, assume_unique=True)
                mask_not_in_impression = np.in1d(impressions, interacted_elements, assume_unique=True)

                interacted_elements = interacted_elements[mask_only_in_impression]
                not_interacted_elements = impressions[~mask_not_in_impression]

                # I append the last interacted elements as first (so I invert the order of relevant_elements!)
                real_recommended = np.flipud(interacted_elements)
                real_recommended = real_recommended.astype(np.int)

                # After the list of num reference, append the other impressions in appearing order
                to_append = not_interacted_elements.astype(np.int)

                recs_tuples.append((self.dictionary_indices.get(i), list(real_recommended) + list(to_append)))

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