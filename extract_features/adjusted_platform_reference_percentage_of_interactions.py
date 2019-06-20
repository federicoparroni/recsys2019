from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions
from preprocess_utils.remove_last_part_of_clk_sessions import remove_last_part_of_clk_sessions

class AdjustedPlatformReferencePercentageOfInteractions(FeatureBase):

    """
    Same feature as PlatformReferencePercentageOfInteractions, but the popularity
    is calculated in a slightly different way as in PersonalizedTopPop.

    user_id | session_id | item_id | adj_percentage_of_total_plat_inter

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'adjusted_platform_reference_percentage_of_interactions'
        super(AdjustedPlatformReferencePercentageOfInteractions, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        # preprocess needed
        df = df.sort_values(by=['user_id','session_id','timestamp','step']).reset_index(drop=True)
        df = remove_last_part_of_clk_sessions(df)
        # get last clickout rows
        last_clickout_indices = find(df)
        clickout_rows = df.loc[last_clickout_indices,
            ['user_id','session_id','platform','reference','impressions']][df.action_type == 'clickout item']

        # get reference rows WITH last clickout
        reference_rows = df[(df.reference.str.isnumeric()==True)]
        # compute popularity WITH last clickout
        df_item_clicks = (
            reference_rows
            .groupby(["reference","platform"])
            .size()
            .reset_index(name="n_interactions_per_item")
        )
        df_item_clicks = df_item_clicks.rename(columns={'reference':'item_id'})
        df_item_clicks['item_id'] = df_item_clicks['item_id'].astype(int)

        df_plat_clicks = (
        reference_rows
            .groupby('platform')
            .size()
            .reset_index(name="n_interactions_per_plat")
        )

        # merge clickout rows expanded with the popularity dataframes
        merged_df = pd.merge(df_item_clicks, df_plat_clicks, how='left', on=['platform']).fillna(0)
        clk_expanded = expand_impressions(clickout_rows)
        feature = pd.merge(clk_expanded, merged_df, how='left', on=['item_id','platform']).fillna(0)
        # compute the percentage of clicks per platfom
        new_col = []
        feature.reference = feature.reference.astype(int)
        feature.item_id = feature.item_id.astype(int)
        for t in tqdm(zip(feature.reference, feature.item_id,
                          feature.n_interactions_per_item, feature.n_interactions_per_plat)):
            if t[0] == t[1]: # è quello cliccato
                if t[3]!=1:
                    percentage_of_total_plat_clk = round(((t[2]-1) * 100.0) / (t[3]-1),5)
                else:
                    percentage_of_total_city_clk = 0
            else: # non è quello cliccato
                if 0 not in [t[2],t[3]] and t[3]!=1:
                    percentage_of_total_plat_clk = round((t[2] * 100.0) / (t[3]-1),5) # tolgo comunque il click per plat
                else:
                    percentage_of_total_plat_clk = 0
            new_col.append(percentage_of_total_plat_clk)
        feature['adj_percentage_of_total_plat_inter'] = new_col
        feature.adj_percentage_of_total_plat_inter = feature.adj_percentage_of_total_plat_inter.astype(float)
        final_feature_reduced = feature[['user_id','session_id','item_id','adj_percentage_of_total_plat_inter']]

        return final_feature_reduced

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = AdjustedPlatformReferencePercentageOfInteractions(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
