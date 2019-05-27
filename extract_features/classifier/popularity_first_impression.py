import math

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
from extract_features.top_pop_per_impression import TopPopPerImpression
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

        feature = TopPopPerImpression(mode=self.mode, cluster=self.cluster).read_feature()
        items = dict()
        for t in tqdm(zip(feature.item_id, feature.top_pop_per_impression), desc="Creating item dict..."):
            items[t[0]] = t[1]
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        target_indices = list(df[df.action_type == "clickout item"].drop_duplicates("session_id", keep="last").index)
        temp = df[df.index.isin(target_indices)]
        first_pop = list()
        max_pop_in_impressions = list()

        for t in tqdm(temp.impressions):
            impressions = list(map(int, t.split("|")))
            fi = impressions[0]
            if fi in items:
                fi_pop = items[fi]
            else:
                fi_pop = 0
            first_pop.append(fi_pop)
            max_pop = fi_pop
            for i in impressions[1:]:
                if i in items:
                    pop = items[i]
                    max_pop = max(pop, max_pop)
            max_pop_in_impressions.append(max_pop)

        temp = temp[["user_id", "session_id"]]
        temp["pop_first_impression"] = first_pop
        temp["max_pop_in_impressions"] = max_pop_in_impressions
        return temp



if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = PopularityFirstImpression(mode=mode, cluster='no_cluster')
    c.save_feature()
