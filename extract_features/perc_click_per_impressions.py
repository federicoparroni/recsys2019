from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions

from collections import Counter

class PercClickPerImpressions(FeatureBase):

    """
    This feature does the following:
    - check how many times an accomodation appears in the impression list
    - check how many times an accomodation has been clicked
    - computes the feature f = (#clicks) / (#times_in_impressions) %

    user_id | session_id | item_id | perc_click_appeared

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'perc_click_per_impressions'
        super(PercClickPerImpressions, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        # get only non-last-clickout clickout rows
        last_clickout_indices = find(df)
        last_clk_removed_df = df.drop(last_clickout_indices)
        reference_rows = last_clk_removed_df[(last_clk_removed_df.reference.str.isnumeric() == True) & (last_clk_removed_df.action_type =='clickout item')][['user_id','session_id','reference','impressions']]

        # get the impressions
        impression_lists = reference_rows.impressions.str.split('|').tolist()
        big_list = [x for l in impression_lists for x in l] # convert multi-dim list in 1-dim list
        c = dict(Counter(big_list)) # count occurence of each accomodation in the impression list

        # create df from dictonary: for each accomodation tells the number of times it appears in impressions
        df_times_in_impressions = pd.DataFrame.from_dict(c, orient='index',columns=['number_of_times_in_impr'])
        df_times_in_impressions['item_id'] = df_times_in_impressions.index.astype(int)
        df_times_in_impressions = df_times_in_impressions.reindex(columns = ['item_id', 'number_of_times_in_impr'])

        # get number of times an accomodation has been clicked
        df_item_clicks = (
            reference_rows
            .groupby(["reference"])
            .size()
            .reset_index(name="n_clickouts")
        )
        df_item_clicks = df_item_clicks.rename(columns={'reference':'item_id'})
        df_item_clicks['item_id'] = df_item_clicks['item_id'].astype(int)

        # merge the two df
        merged = pd.merge(df_times_in_impressions, df_item_clicks, how='left', on=['item_id']).fillna(0)
        merged.n_clickouts = merged.n_clickouts.astype(int)
        merged['perc_click_appeared'] = round((merged.n_clickouts*100)/(merged.number_of_times_in_impr),2)

        # create the feature for each item
        feature_per_item = merged[['item_id','perc_click_appeared']]

        # use the feature for each last clickout
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','impressions']]
        clk_expanded = expand_impressions(clickout_rows)
        final_feature = pd.merge(clk_expanded, feature_per_item, how='left', on=['item_id']).fillna(0)
        final_feature = final_feature[['user_id','session_id','item_id','perc_click_appeared']]

        return final_feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = PercClickPerImpressions(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
