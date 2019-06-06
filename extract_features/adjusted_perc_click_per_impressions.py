from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions

from collections import Counter

class AdjustedPercClickPerImpressions(FeatureBase):

    """
    This feature does the same as PercClickPerImpressions, but computes the
    popularity on the whole df (like PersonalizedTopPop)

    user_id | session_id | item_id | adj_perc_click_appeared

    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'adjusted_perc_click_per_impressions'
        super(AdjustedPercClickPerImpressions, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        # get ALL clickouts
        reference_rows = df[(df.reference.str.isnumeric() == True) & (df.action_type =='clickout item')][['user_id','session_id','reference','impressions']]
        # get last clickout
        last_clickout_indices = find(df)
        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','reference','impressions']]
        clk_expanded = expand_impressions(clickout_rows)

        # get the impressions
        impression_lists = reference_rows.impressions.str.split('|').tolist()
        big_list = [x for l in impression_lists for x in l]
        c = dict(Counter(big_list))

        df_times_in_impressions = pd.DataFrame.from_dict(c, orient='index',columns=['number_of_times_in_impr'])
        df_times_in_impressions['item_id'] = df_times_in_impressions.index.astype(int)
        df_times_in_impressions = df_times_in_impressions.reindex(columns = ['item_id', 'number_of_times_in_impr'])

        feature_times_per_imp = pd.merge(clk_expanded, df_times_in_impressions, how='left', on=['item_id']).fillna(0)
        feature_times_per_imp.number_of_times_in_impr = feature_times_per_imp.number_of_times_in_impr.astype(int)
        feature_times_per_imp = feature_times_per_imp[['user_id', 'session_id','item_id','number_of_times_in_impr']]

        df_item_clicks = (
            reference_rows
            .groupby(["reference"])
            .size()
            .reset_index(name="n_clickouts")
        )
        df_item_clicks = df_item_clicks.rename(columns={'reference':'item_id'})
        df_item_clicks['item_id'] = df_item_clicks['item_id'].astype(int)
        merged = pd.merge(df_times_in_impressions, df_item_clicks, how='left', on=['item_id']).fillna(0)
        merged.n_clickouts = merged.n_clickouts.astype(int)

        final_feature = pd.merge(clk_expanded, merged, how='left', on=['item_id']).fillna(0)
        new_col = []
        final_feature.reference = final_feature.reference.astype(int)
        final_feature.item_id = final_feature.item_id.astype(int)
        for t in tqdm(zip(final_feature.reference, final_feature.item_id,
                     final_feature.number_of_times_in_impr, final_feature.n_clickouts)):
            if t[0]==t[1]: # stessa reference, quindi decremento di 1 sia #click che #imp
                if t[2]!=1:
                    new_col.append(round(((t[3]-1)*100)/(t[2]-1),5))
                else:
                    new_col.append(0)
            else:
                if 0 not in [t[2],t[3]] and t[2]!=1:
                    new_col.append(round(((t[3])*100)/(t[2]-1),5))
                else:
                    new_col.append(0)
        final_feature['adj_perc_click_appeared'] = new_col
        final_feature = final_feature[['user_id','session_id','item_id','adj_perc_click_appeared']]

        return final_feature

if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = AdjustedPercClickPerImpressions(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
