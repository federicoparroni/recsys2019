import math
from extract_features.feature_base import FeatureBase
from extract_features.actions_involving_impression_session import ActionsInvolvingImpressionSession
import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class ActionsCountClassifier(FeatureBase):
    """
    | user_id | session_id | actions count first impression | actions sum other impressions
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'actions_count_classifier'
        super(ActionsCountClassifier, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        feature = ActionsInvolvingImpressionSession(mode=self.mode, cluster=self.cluster).read_feature()

        grouped = feature.groupby("session_id", sort=False)
        fi_list = list()
        others_list = list()
        cols_fi = (
            "actions_involving_impression_session_clickout_item_fi",
            "actions_involving_impression_session_interaction_item_deals_fi",
            "actions_involving_impression_session_interaction_item_image_fi",
            "actions_involving_impression_session_interaction_item_info_fi",
            "actions_involving_impression_session_interaction_item_rating_fi",
            "actions_involving_impression_session_search_for_item_fi",
            "actions_involving_impression_session_no_action_fi")
        cols_others = (
            "actions_involving_impression_session_clickout_item_others",
            "actions_involving_impression_session_interaction_item_deals_others",
            "actions_involving_impression_session_interaction_item_image_others",
            "actions_involving_impression_session_interaction_item_info_others",
            "actions_involving_impression_session_interaction_item_rating_others",
            "actions_involving_impression_session_search_for_item_others",
            "actions_involving_impression_session_no_action_others")

        for name, group in tqdm(grouped, desc="Iteration over sessions..."):

            fi_slice = list(group.iloc[0, 3:])
            others_slice = list(group.iloc[1:, 3:].sum())
            fi_list.append(fi_slice)
            others_list.append(others_slice)

        temp = feature[["user_id", "session_id"]].drop_duplicates("session_id", keep="first")

        for i in range(len(cols_fi)):
            temp[cols_fi[i]] = [x[i] for x in fi_list]

        for i in range(len(cols_others)):
            temp[cols_others[i]] = [x[i] for x in others_list]
        return temp


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = ActionsCountClassifier(mode=mode, cluster='no_cluster')
    c.save_feature()
