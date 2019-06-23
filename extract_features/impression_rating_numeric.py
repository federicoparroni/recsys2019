from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from extract_features.impression_features import ImpressionFeature
import numpy as np


class ImpressionRatingNumeric(FeatureBase):

    """
    impression rating. not categorical but numeric:
    excellent rating: 5
    very good rating: 4
    good rating: 3
    satifactory rating: 2
    no rating: 1
    item_id | rating_numeric
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_rating_numeric'
        super(ImpressionRatingNumeric, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        from extract_features.impression_features import ImpressionFeature
        o = ImpressionFeature(mode=self.mode)
        f = o.read_feature(True)
        f_ratings = f[['item_id', 'propertiesExcellent Rating',
                       'propertiesVery Good Rating',
                       'propertiesGood Rating',
                       'propertiesSatisfactory Rating',
                       ]]
        f_ratings['propertiesNo Rating'] = pd.Series(np.ones(len(f_ratings), dtype=np.uint8),
                                                              index=f_ratings.index)
        df = f_ratings.iloc[:, 1:]
        df['fake'] = pd.Series(np.zeros(len(df), dtype=np.uint8), index=df.index)
        cols = df.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        df = df.reindex(columns=cols)
        dff = df.diff(axis=1).drop(['fake'], axis=1)
        dff = dff.astype(int)
        dff.columns = [5, 4, 3, 2, 1]
        f_ratings = f_ratings.drop(f_ratings.columns[1:], axis=1)
        f_ratings['rating'] = dff.idxmax(axis=1)
        return f_ratings

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = ImpressionRatingNumeric(mode=mode, cluster=cluster)
    c.save_feature()
