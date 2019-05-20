from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()


class TopPopInteractionImagePerImpression(FeatureBase):

    """
    say for each impression of a clickout the popularity of the impression
    | user_id | session_id | reference | popularity
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'top_pop_interaction_image_per_impression'
        super(TopPopInteractionImagePerImpression, self).__init__(
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

            res_df = res_df.rename(columns={'impression_list': 'reference'})
            res_df = res_df.astype({'reference': 'int'})

            return res_df

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        df = df.sort_values(['user_id','session_id','timestamp','step'])
        last_clickout_indices = find_last_clickout_indices(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','action_type','impressions']][df.action_type == 'clickout item']
        # cast the impressions and the prices to lists
        clickout_rows['impression_list'] = clickout_rows.impressions.str.split('|')
        clickout_rows = clickout_rows.drop('impressions', axis=1)

        last_clk_removed_df = df.drop(last_clickout_indices)
        reference_rows = last_clk_removed_df[last_clk_removed_df.reference.astype(str).str.isnumeric()]
        reference_rows = reference_rows[reference_rows['action_type']=='interaction item image']
        reference_rows = reference_rows.sort_index()

        df_item_clicks = (
            reference_rows
            .groupby("reference")
            .size()
            .reset_index(name="n_clicks")
            .transform(lambda x: x.astype(int))
        )

        final_df = expand_impressions(clickout_rows)
        impressions = final_df['reference'].unique().tolist()
        df_item_clicks = df_item_clicks[df_item_clicks['reference'].isin(impressions)]
        final_df['popularity'] = final_df.reference.map(df_item_clicks.set_index('reference')['n_clicks'])
        final_df = final_df.drop(['index', 'action_type'], axis=1)
        final_df.rename(columns={'reference':'item_id'}, inplace=True)
        final_df['popularity'] = final_df['popularity'].fillna(0).astype(int)

        return final_df

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = TopPopInteractionImagePerImpression(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature(one_hot=True))
