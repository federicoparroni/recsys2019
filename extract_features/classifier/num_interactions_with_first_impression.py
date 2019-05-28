import math

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression

tqdm.pandas()


class NumInteractionsWithFirstImpression(FeatureBase):
    """
    count number of interactions with the first impression during the session
    | user_id | session_id | num_interaction_with_first_impression
    num_interaction_with_first_impression is int
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'num_interaction_with_first_impression'
        super(NumInteractionsWithFirstImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        feature = TimesUserInteractedWithImpression(mode=self.mode, cluster=self.cluster).read_feature()
        feature = feature.drop_duplicates("session_id", keep="first")
        feature = feature.drop("item_id", axis=1)
        return feature



if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = NumInteractionsWithFirstImpression(mode=mode, cluster='no_cluster')
    c.save_feature()
