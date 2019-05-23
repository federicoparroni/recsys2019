from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions

class LocationReferencePercentageOfClickouts(FeatureBase):

    """
    This feature for each impression of the clickout says the percentage of
    clickouts made on a city on that impression.
    Example: if the city of the clickout session is Milan and there are overall
    100 clickouts in Milan, for the impression number 3, if it has been clicked
    20 times in Milan, the feature value for the impression 3 will be 0.2 (that means
    that 20% of all the clickouts in Milan have been with impression 3).

    user_id | session_id | item_id | percentage_of_total_city_clk

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'location_reference_percentage_of_clickouts'
        super(LocationReferencePercentageOfClickouts, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        last_clickout_indices = find(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','city','action_type','impressions']]

        last_clk_removed_df = df.drop(last_clickout_indices)
        reference_rows = last_clk_removed_df[(last_clk_removed_df.reference.str.isnumeric() == True) & (last_clk_removed_df.action_type == 'clickout item')]

        df_item_clicks = (
            reference_rows
            .groupby(["reference","city"])
            .size()
            .reset_index(name="n_interactions_per_item")
        )
        df_item_clicks = df_item_clicks.rename(columns={'reference':'item_id'})
        df_item_clicks['item_id'] = df_item_clicks['item_id'].astype(int)

        df_city_clicks = (
        reference_rows
            .groupby('city')
            .size()
            .reset_index(name="n_interactions_per_city")
        )

        final_df = pd.merge(df_item_clicks, df_city_clicks, how='left', on=['city']).fillna(0)

        final_df['percentage_of_total_city_clk'] = 0.0
        for t in zip(final_df.index, final_df.n_interactions_per_item, final_df.n_interactions_per_city):
            percentage_of_total_city_clk = round((t[1] * 100.0) / t[2],2)
            final_df.at[t[0], 'percentage_of_total_city_clk'] = percentage_of_total_city_clk

        feature = final_df[['city','item_id','percentage_of_total_city_clk']]
        clk_expanded = expand_impressions(clickout_rows)
        feature = pd.merge(clk_expanded, feature, how='left', on=['city','item_id']).fillna(0)
        feature = feature[['user_id','session_id','item_id','percentage_of_total_city_clk']]

        return feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = LocationReferencePercentageOfClickouts(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
