from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
from extract_features.impression_features import ImpressionFeature
import numpy as np


class ImpressionStars(FeatureBase):

    """
    Impression star rating in one hot
    item_id | 1_Star | 2_Star | 3_Star | 4_Star | 5_Star
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_stars'
        super(ImpressionStars, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        o = ImpressionFeature(mode='small')
        f = o.read_feature(True)
        feature_stars = f[['item_id','impression_features1 Star', 'impression_features2 Star', 'impression_features3 Star', 'impression_features4 Star', 'impression_features5 Star']]
        feature_stars = feature_stars.rename(columns={'impression_features1 Star':'1_Star', 'impression_features2 Star':'2_Star', 'impression_features3 Star':'3_Star', 'impression_features4 Star':'4_Star','impression_features5 Star':'5_Star'})
        return feature_stars

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = ImpressionStars(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
