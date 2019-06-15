import math

from extract_features.adjusted_location_reference_percentage_of_clickouts import \
    AdjustedLocationReferencePercentageOfClickouts
from extract_features.adjusted_location_reference_percentage_of_interactions import \
    AdjustedLocationReferencePercentageOfInteractions
from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

from extract_features.location_reference_percentage_of_clickouts import LocationReferencePercentageOfClickouts
from extract_features.location_reference_percentage_of_interactions import LocationReferencePercentageOfInteractions
from extract_features.top_pop_interaction_clickout_per_impression import TopPopInteractionClickoutPerImpression
tqdm.pandas()


class LocationReferenceFirstImpression(FeatureBase):
    """
    compress version of location_reference_percentage_of_clickouts without item_id
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'location_reference_first_impression'
        super(LocationReferenceFirstImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        feature = AdjustedLocationReferencePercentageOfClickouts(mode=self.mode, cluster=self.cluster).read_feature()
        feature = feature.drop_duplicates("session_id", keep="first")
        feature = feature.drop(["item_id"], axis=1)

        f2 = AdjustedLocationReferencePercentageOfInteractions(mode=self.mode, cluster=self.cluster).read_feature()
        f2 = f2.drop_duplicates("session_id", keep="first")
        f2 = f2.drop(["item_id"], axis=1)

        feature["location_reference_first_impression_interactions"] = list(f2.percentage_of_total_city_inter)

        return feature


if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()
    c = LocationReferenceFirstImpression(mode=mode, cluster='no_cluster')
    c.save_feature()
