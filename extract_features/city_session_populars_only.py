from extract_features.feature_base import FeatureBase
import data
import pandas as pd



class CitySessionPopularsOnly(FeatureBase):
    """
    say for each session the city of the platform if is one of the most populars. Otherwise, category is 'not_popular_city'
    | user_id | session_id | city_popular
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'city_populars_only_session'
        columns_to_onehot = [('city_popular', 'single')]
        super(CitySessionPopularsOnly, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)

    def extract_feature(self):
        def get_popular_cities(city):
            dict_c = {}

            for c in city:
                if c in dict_c.keys():
                    dict_c[c] += 1
                else:
                    dict_c[c] = 1

            items = sorted(dict_c.items(), key=lambda x: x[1], reverse=True)
            items = [x[0] for x in items]

            return items[:100]#TODO: formalize

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        user_ids = df.user_id.values
        sessions = df.session_id.values
        city = df.city.values

        popular_cities = get_popular_cities(city)

        city_new = []
        for c in city:
            if c in popular_cities:
                city_new.append(c)
            else:
                city_new.append('not_popular_city')

        final_df = pd.DataFrame({'user_id': user_ids, 'session_id': sessions, 'city_popular': city_new})
        final_df.drop_duplicates(inplace=True, subset=['user_id', 'session_id'])
        return final_df


if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = CitySessionPopularsOnly(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature(one_hot=True))
