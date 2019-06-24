from extract_features.feature_base import FeatureBase
import data
import pandas as pd


class CitySession(FeatureBase):
    """
    say for each session the platform nationality and the city of the platform
    | user_id | session_id | platform

    WARNING:
    Do not use single encoding (binary could be an option) since there are too many values.
    Works with LightGBM & Catboost
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'city_session'
        columns_to_onehot = [('city', 'binary')]
        super(CitySession, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        user_ids = df.user_id.values
        sessions = df.session_id.values
        city = df.city.values

        final_df = pd.DataFrame({'user_id': user_ids, 'session_id': sessions, 'city': city})
        final_df.drop_duplicates(inplace=True, subset=['user_id', 'session_id'])
        return final_df


if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = CitySession(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature(one_hot=True))
