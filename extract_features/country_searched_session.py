from extract_features.feature_base import FeatureBase
import data
import pandas as pd


class CountrySearchedSession(FeatureBase):
    """
    say for each session the country of the city searched
    | user_id | session_id | country_searched

    WARNING:
    Do not use single encoding (binary could be an option) since there are too many values.
    Works with LightGBM & Catboost
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'country_searched_session'
        columns_to_onehot = [('country_searched', 'single')]
        super(CountrySearchedSession, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        user_ids = df.user_id.values
        sessions = df.session_id.values
        country_searched = [x.split(',')[1][1:] for x in df.city.values]

        final_df = pd.DataFrame({'user_id': user_ids, 'session_id': sessions, 'country_searched': country_searched})
        final_df.drop_duplicates(inplace=True, subset=['user_id', 'session_id'])
        return final_df


if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = CountrySearchedSession(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature(one_hot=True))
