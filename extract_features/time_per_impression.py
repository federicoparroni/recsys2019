from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions

class TimePerImpression(FeatureBase):

    """
    say for each impression of a clickout the total time spent on it
    | user_id | session_id | item_id | impression_time
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'time_per_impression'
        super(TimePerImpression, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        def convert_and_add_pos(df):
            df_t = expand_impressions(df)
            df['index'] = df.index
            df = pd.merge(df_t, df, how='left', on=['index', 'user_id', 'session_id','action_type'], suffixes=('', '_y'))
            df = df.drop('time_per_impression_y', axis=1)
            df['item_pos'] = df.apply(lambda x: (x['impression_list'].index(str(x['item_id']))) + 1, axis=1)
            df = df.drop(['impression_list', 'index'], axis=1)
            return df

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        df = df.sort_values(['user_id','session_id','timestamp','step']).reset_index(drop=True)
        df['time_per_impression'] = df['timestamp'].shift(-1)-df['timestamp']

        last_clickout_indices = find(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','action_type','impressions']][df.action_type == 'clickout item']
        clickout_rows['impression_list'] = clickout_rows.impressions.str.split('|')
        clickout_rows['time_per_impression'] = [[0]*25 for x in range(len(clickout_rows.index))]

        last_clk_removed_df = df.drop(last_clickout_indices)
        reference_rows = last_clk_removed_df[last_clk_removed_df.reference.astype(str).str.isnumeric()]
        reference_rows = reference_rows.drop('action_type',axis=1)
        reference_rows = reference_rows[reference_rows.user_id.isin(clickout_rows.user_id) & reference_rows.session_id.isin(clickout_rows.session_id)]

        j = 0
        clickout_indices = clickout_rows.index.values
        clickout_user = clickout_rows.at[clickout_indices[j], 'user_id']
        clickout_session = clickout_rows.at[clickout_indices[j], 'session_id']
        for t in tqdm(zip(reference_rows.index, reference_rows.time_per_impression, reference_rows.user_id, reference_rows.session_id, reference_rows.reference)):
            if t[0] >= clickout_indices[-1]:
                break
            # find the next clickout index
            while t[0] > clickout_indices[j]:
                j += 1
                clickout_user = clickout_rows.at[clickout_indices[j], 'user_id']
                clickout_session = clickout_rows.at[clickout_indices[j], 'session_id']

            # check if row and next_clickout are in the same session
            if t[2] == clickout_user and t[3] == clickout_session:
                try:
                    ref_idx = clickout_rows.at[clickout_indices[j], 'impression_list'].index(t[4])
                    feature_list = clickout_rows.at[clickout_indices[j], 'time_per_impression']
                    feature_list[ref_idx] += t[1]
                except:
                    pass

        final_df = convert_and_add_pos(clickout_rows)
        final_df['impression_time'] = final_df.apply(lambda x: list(x['time_per_impression'])[int(x['item_pos']) - 1], axis=1)
        final_df = final_df[['user_id','session_id','item_id','impression_time']]
        final_df['impression_time'] = final_df['impression_time'].astype(int)
        return final_df

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = TimePerImpression(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
