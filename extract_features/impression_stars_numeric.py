from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
from extract_features.impression_features import ImpressionFeature
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
import numpy as np


class ImpressionStarsNumeric(FeatureBase):

    """
    Impression star rating in numeric form (default 0 for missing star rating):

    user_id | session_id | item_id | stars

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_stars_numeric'
        super(ImpressionStarsNumeric, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        last_clickout_indices = find(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','impressions']]
        clk_expanded = expand_impressions(clickout_rows)

        o = ImpressionFeature(mode=self.mode)
        f = o.read_feature(True) # get the accomodation's df
        feature_stars = f[['item_id','properties1 Star', 'properties2 Star', 'properties3 Star', 'properties4 Star', 'properties5 Star']]
        # remap the name
        feature_stars = feature_stars.rename(columns={'properties1 Star':'1', 'properties2 Star':'2', 'properties3 Star':'3', 'properties4 Star':'4','properties5 Star':'5'})
        # set default 0 Stars for those for which the feature is missing
        feature_stars['0'] = pd.Series(np.ones(len(feature_stars), dtype=np.uint8),index=feature_stars.index)
        feature_stars['stars'] = feature_stars[['5','4','3','2','1','0']].idxmax(axis=1)
        feature_stars_restricted = feature_stars[['item_id', 'stars']]
        final_feature = pd.merge(clk_expanded, feature_stars_restricted, how='left', on=['item_id']).fillna(1)
        final_feature['stars'] = final_feature['stars'].astype(int)
        final_feature['stars'] = final_feature['stars'].replace(0, -1)
        return final_feature[['user_id','session_id','item_id','stars']]

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = ImpressionStarsNumeric(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
