import math
from extract_features.feature_base import FeatureBase

import data
import pandas as pd
from tqdm.auto import tqdm

from extract_features.user_feature import UserFeature

tqdm.pandas()


class UserFeatureFirstImpression(FeatureBase):
    """
    See impression_price_info_session
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'user_feature_first_impression'
        super(UserFeatureFirstImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        feature = UserFeature(mode=self.mode, cluster=self.cluster).read_feature()
        feature = feature.drop_duplicates("session_id", keep="first")
        feature = feature.drop(["item_id"], axis=1)
        return feature


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = UserFeatureFirstImpression(mode=mode, cluster='no_cluster')
    c.save_feature()
