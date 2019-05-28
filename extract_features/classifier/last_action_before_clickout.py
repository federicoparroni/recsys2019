from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


class LastActionBeforeClickout(FeatureBase):

    """
    last action occourring before each last clickout:
    | user_id | session_id | last_action
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'last_action_before_clickout'
        columns_to_onehot = [('last_action', 'single')]
        super(LastActionBeforeClickout, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        temp = df[df.action_type == "clickout item"]
        temp = temp.drop_duplicates("session_id", keep="last")
        temp = temp[["user_id", "session_id", "step", "action_type"]]
        actions = list()
        for index, row in tqdm(temp.iterrows(), desc="Scanning clickouts"):
            if index > 0:
                if int(row.step) == 1:
                    actions.append("no_action")
                else:
                    actions.append(df.loc[index - 1, "action_type"])
            else:
                actions.append("no_action")

        temp = temp.drop(["action_type", "step"], axis=1)
        temp["last_action"] = actions
        s = temp
        return s

if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = LastActionBeforeClickout(mode=mode, cluster='no_cluster')
    c.save_feature()
