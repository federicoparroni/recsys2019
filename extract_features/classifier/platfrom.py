import math

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class Platform(FeatureBase):
    """
    count number of interactions with the first impression during the session
    | user_id | session_id | first_impression_price | price_position_in_ascending_order
    num_interaction_with_first_impression is int
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'platform'
        columns_to_onehot = [('platform', 'single')]
        super(Platform, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        target_indices = list(df[df.action_type == "clickout item"].drop_duplicates("session_id", keep="last").index)

        values = list(df.loc[target_indices, "platform"])

        if len(target_indices) != len(values):
            print("Something went wrong, blame Piccio")
            exit(69)

        temp = df.loc[target_indices, ["user_id", "session_id"]]
        temp["platform"] = values
        return temp


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = Platform(mode=mode, cluster='no_cluster')
    c.save_feature()
