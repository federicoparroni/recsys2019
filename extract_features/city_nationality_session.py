from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

class CityPlatformSession(FeatureBase):

    """
    say for each session the platform nationality and the city of the platform
    | user_id | session_id | platform_nationality | city
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'city_platform_session'
        super(CityPlatformSession, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        def find_last_clickout_indices(df):
            indices = []
            cur_ses = ''
            cur_user = ''
            temp_df = df[df.action_type == 'clickout item'][['user_id', 'session_id', 'action_type']]
            for idx in tqdm(temp_df.index.values[::-1]):
                ruid = temp_df.at[idx, 'user_id']
                rsid = temp_df.at[idx, 'session_id']
                if (ruid != cur_user or rsid != cur_ses):
                    indices.append(idx)
                    cur_user = ruid
                    cur_ses = rsid
            return indices[::-1]

        df = data.full_df()
        df = df.sort_index()
        df['time_per_impression'] = df['timestamp'].shift(-1)-df['timestamp']
        df = df.drop('timestamp', axis=1)

        last_clickout_indices = find_last_clickout_indices(df)
        target_rows = df.iloc[last_clickout_indices, :]

        user_ids = target_rows.user_id.values
        sessions = target_rows.session_id.values
        nationalities = target_rows.platform.values
        city = target_rows.city.values

        final_df = pd.DataFrame({'user_id': user_ids, 'session_id': sessions, 'platform_nationality': nationalities, 'city': city})
        return final_df

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = CityPlatformSession(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature(one_hot=False))
