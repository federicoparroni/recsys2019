import math
from extract_features.impression_rating_numeric import ImpressionRatingNumeric
from extract_features.impression_stars_numeric import ImpressionStarsNumeric
from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class StarsRatingsFirstImpression(FeatureBase):
    """
    | user_id | session_id | popularity_first_impression | max_popularity_other_impressions
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'starts_ratings_first_impression'
        super(StarsRatingsFirstImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        feature = ImpressionRatingNumeric(mode=self.mode, cluster=self.cluster).read_feature()
        items = dict()
        for t in tqdm(zip(feature.item_id, feature.rating), desc="Creating item dict..."):
            items[t[0]] = t[1]
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        target_indices = list(df[df.action_type == "clickout item"].drop_duplicates("session_id", keep="last").index)
        temp = df[df.index.isin(target_indices)].sort_values("session_id")
        first_ratings = list()

        for t in tqdm(temp.impressions):
            impressions = list(map(int, t.split("|")))
            fi = impressions[0]
            if fi in items:
                fi_rating = items[fi]
            else:
                fi_rating = 0
            first_ratings.append(fi_rating)

        temp = temp[["user_id", "session_id"]]
        temp["first_impression_rating"] = first_ratings

        feature = ImpressionStarsNumeric(mode=self.mode, cluster=self.cluster).read_feature()
        feature = feature.drop_duplicates("session_id", keep="first").sort_values("session_id")

        temp["first_impression_stars"] = list(feature.stars)
        return temp


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = StarsRatingsFirstImpression(mode=mode, cluster='no_cluster')
    c.save_feature()
