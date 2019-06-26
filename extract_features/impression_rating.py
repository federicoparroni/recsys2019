from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from extract_features.impression_features import ImpressionFeature
import numpy as np


class ImpressionRating(FeatureBase):

    """
    impression features
    item_id | rating
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_rating'
        columns_to_onehot = [('rating', 'single')]
        super(ImpressionRating, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):
        from extract_features.impression_features import ImpressionFeature
        o = ImpressionFeature(mode=self.mode)
        f = o.read_feature(True)
        f_ratings = f[['item_id', 'impression_featuresExcellent Rating',
                       'impression_featuresVery Good Rating',
                       'impression_featuresGood Rating',
                       'impression_featuresSatisfactory Rating',
                       ]]
        f_ratings['impression_featuresNo Rating'] = pd.Series(np.ones(len(f_ratings), dtype=np.uint8),
                                                              index=f_ratings.index)
        df = f_ratings.iloc[:, 1:]
        df['fake'] = pd.Series(np.zeros(len(df), dtype=np.uint8), index=df.index)
        cols = df.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        df = df.reindex(columns=cols)
        dff = df.diff(axis=1).drop(['fake'], axis=1)
        dff = dff.astype(int)
        dff.columns = ['Excellent Rating', 'Very Good Rating',
                       'Good Rating', 'Satisfactory Rating', 'No Rating']
        dff.idxmax(axis=1)
        f_ratings = f_ratings.drop(f_ratings.columns[1:], axis=1)
        f_ratings['rating'] = dff.idxmax(axis=1)
        return f_ratings

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = ImpressionRating(mode=mode, cluster=cluster)
    c.save_feature()
