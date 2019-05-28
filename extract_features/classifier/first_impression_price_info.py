import math
from extract_features.feature_base import FeatureBase
from extract_features.impression_price_info_session import ImpressionPriceInfoSession

import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class FirstImpressionPriceInfo(FeatureBase):
    """
    See impression_price_info_session
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'first_impression_price_info'
        super(FirstImpressionPriceInfo, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        feature = ImpressionPriceInfoSession(mode=self.mode, cluster=self.cluster).read_feature()
        feature = feature.drop_duplicates("session_id", keep="first")
        feature = feature.drop(["item_id"], axis=1)
        return feature


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = FirstImpressionPriceInfo(mode=mode, cluster='no_cluster')
    c.save_feature()
