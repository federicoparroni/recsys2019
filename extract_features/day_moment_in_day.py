import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
from extract_features.feature_base import FeatureBase
from preprocess_utils.last_clickout_indices import find
import data
import numpy as np
import pytz


class DayOfWeekAndMomentInDay(FeatureBase):

    """
    for every user_id and session_id the day of the week and the moment in the day of last clickout .
    the timestamp is shifted in the timezone of the platform.
    SDT has been considered for Austral Emisphere platforms since all timestamps are in November.

    user_id | session_id | day | moment

    decisions:
        RU --> take Moscow timezone
        AU --> take Adelaide timezone
        CA --> take Winnipeg timezone
        US --> take Oklahoma City timezone
        BR --> take Rio de Janeiro timezone


    day : The day of the week with Monday=0, Sunday=6
    moment: [00.00 : 8:00) --> N  (night)
            [8:00 : 13:00) --> M  (morning)
            [13:00 : 19:00) --> A (afternoon)
            [19:00 : 00:00) --> E (evening)
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'day_of_week_and_moment_in_day'
        super(DayOfWeekAndMomentInDay, self).__init__(
            name=name, mode=mode, cluster=cluster, columns_to_onehot=[('day', 'single'), ('moment', 'single')])

    def extract_feature(self):

        def func(x):

            def extract_daytime(timestamp, platform):
                res = np.empty(len(timestamp), dtype='datetime64[s]')
                unique_platforms = x['platform'].unique()
                dict_row_platform = {

                    'AU': 3,
                    'CA': 11,
                    'RU': 1,
                    'BR': 1,
                    'US': 17

                }

                list_of_common_platforms = [i for i in unique_platforms if i not in dict_row_platform.keys()]

                for i in list_of_common_platforms:
                    dict_row_platform[i] = 0

                dict_row_platform['GB'] = dict_row_platform.pop('UK')
                dict_row_platform['ET'] = dict_row_platform.pop('AA')

                austral_emisphere = ['AU', 'MX', 'CL', 'AR', 'ID', 'NZ', 'EC', 'BR']

                for i in tqdm(range(len(timestamp))):
                    ts = timestamp[i]
                    p = platform[i]
                    if p == 'UK':
                        p = 'GB'
                    elif p == 'AA':
                        p = 'ET'

                    if p in austral_emisphere:
                        bool_amb = True

                    else:
                        bool_amb = False

                    zone = pytz.country_timezones(p)[dict_row_platform[p]]

                    timeznd = pd.to_datetime(ts).tz_localize(zone, ambiguous=np.array(bool_amb))

                    res[i] = timeznd

                return res

            return extract_daytime(pd.to_datetime(x['timestamp'], unit='s', origin='unix').values,
                                   x['platform'].values).astype('datetime64[s]')

        def get_moment_in_the_day(x):
            if (0 <= x.hour) & (x.hour < 8):
                return 'N'
            elif (8 <= x.hour) & (x.hour < 13):
                return 'M'
            elif (13 <= x.hour) & (x.hour < 19):
                return 'A'
            elif (19 <= x.hour) & (x.hour < 24):
                return 'E'

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df_indices = find(pd.concat([train, test]))
        df = pd.concat([train, test]).loc[df_indices]

        df['day'] = func(df)
        df['moment'] = df['day'].progress_apply(lambda x: get_moment_in_the_day(x))
        df['day'] = df['day'].progress_apply(lambda x: pd.to_datetime(x).dayofweek)

        return df.drop(
            columns=['action_type', 'reference', 'impressions', 'prices', 'city', 'device', 'step', 'current_filters',
                     'timestamp', 'platform', 'frequence'])

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection
    cluster = cluster_selection()
    mode = mode_selection()
    c = DayOfWeekAndMomentInDay(mode=mode, cluster=cluster)
    c.save_feature()
    print(c.read_feature())