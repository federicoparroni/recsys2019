import math

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm


tqdm.pandas()


class NumImpressionsInClickout(FeatureBase):
    """
    Number of impressions in clickout list
    Idea - extreme scenario: if it is 1, the classifier must learn to predict always a class 1
         - standard scenario: the probability of clicking the first impression increase if the number of impressions shown decrease
    | user_id | session_id | num_impressions_in_clickout

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'num_impression_in_clickout'
        super(NumImpressionsInClickout, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        temp = df[df.action_type == "clickout item"]
        temp = temp.drop_duplicates("session_id", keep="last")
        num_impressions = []
        for t in tqdm(temp.impressions):
            num_impressions.append(len(t.split("|")))

        temp = temp[["user_id", "session_id"]]
        if len(temp) != len(num_impressions):
            print("Piccio svegliati")
            exit(-1)

        temp["num_impressoins_in_clickout"] = num_impressions
        return temp


if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = NumImpressionsInClickout(mode=mode, cluster=cluster)
    c.save_feature()
