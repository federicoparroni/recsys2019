from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()


class TimeImpressionLabel(FeatureBase):

    """
    say for each impression of a clickout the total time spent on it
    | user_id | session_id | item_id | time_per_impression
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'time_per_impression'
        super(TimeImpressionLabel, self).__init__(
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

        def expand_impressions(df):
            res_df = df.copy()
            res_df = res_df.reset_index()
            res_df = pd.DataFrame({
                col: np.repeat(res_df[col].values, res_df.impression_list.str.len())
                for col in res_df.columns.drop('impression_list')
            }
            ).assign(**{'impression_list': np.concatenate(res_df.impression_list.values)})[res_df.columns]

            res_df = res_df.rename(columns={'impression_list': 'item_id'})
            res_df = res_df.astype({'item_id': 'int'})

            return res_df

        def convert_and_add_pos(df):
            df_t = expand_impressions(df)
            df['index'] = df.index
            df = pd.merge(df_t, df, how='left', on=['index', 'user_id', 'session_id','action_type'], suffixes=('', '_y'))
            df = df.drop('time_per_impression_y', axis=1)
            df['item_pos'] = df.progress_apply(lambda x: (x['impression_list'].index(str(x['item_id']))) + 1, axis=1)
            df = df.drop(['impression_list', 'index'], axis=1)
            return df

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        df = df.sort_index()
        df['time_per_impression'] = df['timestamp'].shift(-1)-df['timestamp']
        df = df.drop('timestamp', axis=1)

        last_clickout_indices = find_last_clickout_indices(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','action_type','impressions']][df.action_type == 'clickout item']
        # cast the impressions and the prices to lists
        clickout_rows['impression_list'] = clickout_rows.impressions.str.split('|')
        clickout_rows = clickout_rows.drop('impressions', axis=1)
        clickout_rows['time_per_impression'] = [[0]*25 for x in range(len(clickout_rows.index))]

        reference_rows = df[['user_id','session_id','reference','action_type','time_per_impression']]
        reference_rows = reference_rows[df.reference.str.isnumeric() & (df.action_type != 'clickout item')]
        reference_rows = reference_rows.drop('action_type',axis=1)
        reference_rows = reference_rows.sort_index()

        j = 0
        clickout_indices = clickout_rows.index.values
        clickout_user = clickout_rows.at[clickout_indices[j], 'user_id']
        clickout_session = clickout_rows.at[clickout_indices[j], 'session_id']
        for idx,row in tqdm(reference_rows.iterrows()):
            # if the current index is over the last clickout, break
            if idx >= clickout_indices[-1]:
                break
            # find the next clickout index
            while idx > clickout_indices[j]:
                j += 1
                clickout_user = clickout_rows.at[clickout_indices[j], 'user_id']
                clickout_session = clickout_rows.at[clickout_indices[j], 'session_id']

            # check if row and next_clickout are in the same session
            if row.user_id == clickout_user and row.session_id == clickout_session:
                try:
                    ref_idx = clickout_rows.at[clickout_indices[j], 'impression_list'].index(row.reference)
                    feature_list = clickout_rows.at[clickout_indices[j], 'time_per_impression']
                    feature_list[ref_idx] += row['time_per_impression']
                except:
                    pass
        final_df = convert_and_add_pos(clickout_rows)
        final_df['impression_time'] = final_df.apply(lambda x: list(x['time_per_impression'])[int(x['item_pos']) - 1], axis=1)
        final_df = final_df.drop(['time_per_impression','item_pos','action_type'], axis=1)
        return final_df

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = TimeImpressionLabel(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature(one_hot=True))
