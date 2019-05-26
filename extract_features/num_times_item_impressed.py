from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from preprocess_utils.last_clickout_indices import find
from preprocess_utils.last_clickout_indices import expand_impressions

from collections import Counter

class NumTimesItemImpressed(FeatureBase):
    """
    This feature tells for every item in the impressions of a clickout how many
    times that item has been previously impressed.

    user_id | session_id | item_id | num_times_item_impressed

    """
    def __init__(self, mode, cluster='no_cluster'):
        name = 'num_times_item_impressed'
        super(NumTimesItemImpressed, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])

        last_clickout_indices = find(df)
        last_clk_removed_df = df.drop(last_clickout_indices)
        reference_rows = last_clk_removed_df[(last_clk_removed_df.reference.str.isnumeric() == True) & (last_clk_removed_df.action_type =='clickout item')][['user_id','session_id','reference','impressions']]

        clickout_rows = df.loc[last_clickout_indices, ['user_id','session_id','impressions']]
        clk_expanded = expand_impressions(clickout_rows)

        impression_lists = reference_rows.impressions.str.split('|').tolist()
        big_list = [x for l in impression_lists for x in l] # flatten multi dim list in 1-dim list :)
        c = dict(Counter(big_list)) # count occurrence of each item_id in the impressions

        df_times_in_impressions = pd.DataFrame.from_dict(c, orient='index',columns=['num_times_item_impressed'])
        df_times_in_impressions['item_id'] = df_times_in_impressions.index.astype(int)
        df_times_in_impressions = df_times_in_impressions.reindex(columns = ['item_id', 'num_times_item_impressed'])
        df_times_in_impressions = df_times_in_impressions.sort_values(by=['item_id']).reset_index(drop=True)

        feature = pd.merge(clk_expanded, df_times_in_impressions, how='left', on=['item_id']).fillna(0)
        feature.num_times_item_impressed = feature.num_times_item_impressed.astype(int)

        return feature[['user_id','session_id','item_id','num_times_item_impressed']]


if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = NumTimesItemImpressed(mode, cluster)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
