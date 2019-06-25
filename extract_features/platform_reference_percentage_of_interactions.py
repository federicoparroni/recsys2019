from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
from preprocess_utils.remove_last_part_of_clk_sessions import remove_last_part_of_clk_sessions

class PlatformReferencePercentageOfInteractions(FeatureBase):

    """
    This feature for each impression of the clickout says the percentage of
    interactions made on a platform on that impression.
    Example: if the platform of the clickout session is IT and there are overall
    100 interactions in IT, for the impression number 3, if it has been iteracted
    20 times in IT, the feature value for the impression 3 will be 0.2 (that means
    that 20% of all the interactions in IT have been with impression 3).

    user_id | session_id | item_id | percentage_of_total_plat_inter

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'platform_reference_percentage_of_interactions'
        super(PlatformReferencePercentageOfInteractions, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        last_clickout_indices = find(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','platform','action_type','impressions']]

        last_clk_removed_df = df.drop(last_clickout_indices)
        reference_rows = last_clk_removed_df[(last_clk_removed_df.reference.str.isnumeric() == True)]

        df_item_clicks = (
            reference_rows
            .groupby(["reference","platform"])
            .size()
            .reset_index(name="n_interactions_per_item")
        )
        df_item_clicks = df_item_clicks.rename(columns={'reference':'item_id'})
        df_item_clicks['item_id'] = df_item_clicks['item_id'].astype(int)

        df_city_clicks = (
        reference_rows
            .groupby('platform')
            .size()
            .reset_index(name="n_interactions_per_plat")
        )

        final_df = pd.merge(df_item_clicks, df_city_clicks, how='left', on=['platform']).fillna(0)

        final_df['percentage_of_total_plat_inter'] = 0.0
        for t in zip(final_df.index, final_df.n_interactions_per_item, final_df.n_interactions_per_plat):
            percentage_of_total_plat_inter = round((t[1] * 100.0) / t[2],2)
            final_df.at[t[0], 'percentage_of_total_plat_inter'] = percentage_of_total_plat_inter

        feature = final_df[['platform','item_id','percentage_of_total_plat_inter']]
        clk_expanded = expand_impressions(clickout_rows)
        feature = pd.merge(clk_expanded, feature, how='left', on=['platform','item_id']).fillna(0)
        feature = feature[['user_id','session_id','item_id','percentage_of_total_plat_inter']]

        return feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = PlatformReferencePercentageOfInteractions(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
