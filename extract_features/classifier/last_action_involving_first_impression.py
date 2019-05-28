from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class LastActionInvolvingFirstImpressions(FeatureBase):

    """
    Check if the last action before clickout is refeered to the first impression or not
    | user_id | session_id | flag
    flag = 1 means last action on first impression, 0 otherwise
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'last_action_involving_first_impression'
        super(LastActionInvolvingFirstImpressions, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        temp = df[df.action_type == "clickout item"]
        temp = temp.drop_duplicates("session_id", keep="last")
        temp = temp[["user_id", "session_id", "step", "impressions"]]
        actions = list()
        for index, row in tqdm(temp.iterrows(), desc="Scanning clickouts"):
            if index > 0:
                if int(row.step) == 1:
                    actions.append(0)
                else:
                    impression = list(map(int, row.impressions.split("|")))[0]
                    reference = df.loc[index - 1, "reference"]
                    if (type(reference) == str) and (reference.isdigit()) and (int(reference) == impression):
                        actions.append(1)
                    else:
                        actions.append(0)
            else:
                actions.append(0)

        temp = temp.drop(["step", "impressions"], axis=1)
        temp["last_action_involving_first_impression"] = actions
        return temp

if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = LastActionInvolvingFirstImpressions(mode=mode, cluster='no_cluster')
    c.save_feature()
