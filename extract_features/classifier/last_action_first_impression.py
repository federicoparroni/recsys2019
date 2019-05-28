import math

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

from extract_features.last_action_involving_impression import LastInteractionInvolvingImpression
from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression

tqdm.pandas()


class LastActionFirstImpression(FeatureBase):
    """
    Last interaction with the first impression
    | user_id | session_id | last_action
    num_interaction_with_first_impression is int
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'last_action_first_impression'
        columns_to_onehot = [('last_action_first_impression', 'single')]
        super(LastActionFirstImpression, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        temp = df[df.action_type == "clickout item"]
        temp = temp.drop_duplicates("session_id", keep="last")
        first_impressoins = dict()
        for t in tqdm(zip(temp.session_id, temp.impressions)):
            first_impression = list(map(int, t[1].split("|")))[0]
            first_impressoins[t[0]] = first_impression


        #init dict with no action for every session
        last_actions = dict()
        for t in temp.session_id:
            last_actions[t] = "no_action"

        #remove bias action (clickout) from the list of actions
        bias_index = df[df.action_type == "clickout item"].drop_duplicates("session_id", keep="last").index
        temp = df[(df.session_id.isin(list(first_impressoins.keys()))) & (~df.index.isin(bias_index))]
        current_session = list(temp.session_id)[0]
        last_session = list(temp.session_id)[-1]
        last_actions[last_session] = "no_action"
        for t in tqdm(zip(temp.session_id, temp.action_type, temp.reference)):
            if t[0] != current_session and current_session not in last_actions:
                last_actions[current_session] = "no_action"
            current_session = t[0]
            if t[2].isdigit() and int(t[2]) == first_impressoins[t[0]]:
                last_actions[t[0]] = t[1]

        temp = df[df.action_type == "clickout item"]
        temp = temp.drop_duplicates("session_id", keep="last")
        temp = temp[["user_id", "session_id"]]
        if len(temp) != len(last_actions):
            print("Piccio svegliati")
            exit(-1)

        temp["last_action_first_impression"] = list(last_actions.values())
        return temp

        # feature = LastInteractionInvolvingImpression(mode=self.mode, cluster=self.cluster).read_feature()
        # feature = feature.drop_duplicates("session_id", keep="first")
        # feature = feature.drop("item_id", axis=1)
        # return feature


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = LastActionFirstImpression(mode=mode, cluster='no_cluster')
    c.save_feature()
