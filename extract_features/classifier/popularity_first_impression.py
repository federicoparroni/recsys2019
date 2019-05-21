import math

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

from extract_features.top_pop_interaction_image_per_impression import TopPopInteractionImagePerImpression

tqdm.pandas()


class PopularityFirstImpression(FeatureBase):
    """
    | user_id | session_id | popularity_first_impression | max_popularity_other_impressions
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'popularity_first_impression'
        super(PopularityFirstImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        feature = TopPopInteractionImagePerImpression(mode=self.mode, cluster=self.cluster).read_feature()

        session = feature.session_id.values[0]
        max_pop = 0
        others_pop = list()
        for t in tqdm(zip(feature.session_id, feature.popularity)):
            if t[0] != session:
                session = t[0]
                others_pop.append(max_pop)
                max_pop = 0
            else:
                max_pop = max((t[1], max_pop))
        others_pop.append(max_pop)



        feature = feature.drop_duplicates("session_id", keep="first")
        feature = feature.drop("item_id", axis=1)
        feature["max_popularity_other_impressions"] = others_pop
        return feature



if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = PopularityFirstImpression(mode=mode, cluster='no_cluster')
    c.save_feature()
