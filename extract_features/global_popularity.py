import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import pandas as pd

class GlobalPopularity(FeatureBase):

    """
    Compute the popularity of a reference, that's it, no tricks. The popularity is calculated in the full df.
    | reference | popularity
    popularity is a positive number
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'global_popularity'
        columns_to_onehot = []

        super().__init__(name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)


    def extract_feature(self):
        df = data.full_df()

        # count the numeric references
        res_df = df[(df.reference.str.isnumeric() == True) & (df.action_type != 'clickout item')]
        res_df = res_df.astype({'reference':'int'})
        res_df = res_df[['reference','frequence']].groupby('reference').sum()

        return res_df.rename(columns={'frequence':'glob_popularity'}).reset_index()

    
    def join_to(self, df):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature()
        # set the reference column to int
        res_df = df.copy()
        res_df.loc[(res_df.reference.str.isnumeric() != True) | (res_df.action_type == 'clickout item'), 'reference'] = 0
        res_df = res_df.astype({'reference': 'int'}).reset_index()
        res_df = res_df.merge(feature_df, how='left', left_on='reference', right_on='reference').set_index('index')
        res_df['reference'] = df['reference']
        return res_df.fillna(0).astype({'glob_popularity': 'int'})


if __name__ == '__main__':
    import utils.menu as menu
    
    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = GlobalPopularity(mode=mode, cluster=cluster)
    #c.save_feature()
    print(c.read_feature())
