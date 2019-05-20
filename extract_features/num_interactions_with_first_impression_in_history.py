import math

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression

tqdm.pandas()


class NumInteractionsWithFirstImpressionInHistory(FeatureBase):
    """
    count number of interactions with the first impression in all the session of a certain user
    | user_id | session_id | num_interaction_with_first_impression
    num_interaction_with_first_impression is int
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'num_interaction_with_first_impression_in_history'
        super(NumInteractionsWithFirstImpressionInHistory, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        feature = TimesUserInteractedWithImpression(mode=self.mode, cluster=self.cluster).read_feature()
        temp = feature.copy()
        feature = feature.drop_duplicates("session_id", keep="first")
        feature.drop("n_times_clicked_before_clk", axis=1)
        temp = temp.groupby(["user_id", "item_id"])["n_times_clicked_before_clk"].sum().reset_index()
        feature = feature.merge(temp, how='inner')
        feature = feature.drop("item_id", axis=1)
        feature.columns = ["user_id", "session_id", "count_interactions_first_impression_history"]
        return feature



if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = NumInteractionsWithFirstImpressionInHistory(mode=mode, cluster='no_cluster')
    c.save_feature()
