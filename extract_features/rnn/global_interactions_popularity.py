import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import pandas as pd

class GlobalInteractionsPopularity(FeatureBase):

    """
    Compute the popularity of a reference by means of the number of times it has been interacted (in whatever action).
    The popularity is calculated in the train df.
    | item_id | popularity
    popularity is a positive number
    """

    def __init__(self, mode, cluster):
        name = 'global_interactions_popularity'
        columns_to_onehot = []

        super().__init__(name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot)


    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        del train
        del test

        # count the numeric references
        res_df = df[(df.reference.str.isnumeric() == True)]
        res_df = res_df.astype({'reference':'int'})
        res_df = res_df[['reference','frequence']].groupby('reference').sum()

        res_df = res_df.reset_index()
        return res_df.rename(columns={'reference': 'item_id', 'frequence': 'glob_inter_popularity'})

    
    def join_to(self, df):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature()
        # set the reference column to int
        res_df = df.copy()
        res_df.loc[(res_df.reference.str.isnumeric() != True) | (res_df.action_type == 'clickout item'), 'reference'] = 0
        res_df = res_df.astype({'reference': 'int'}).reset_index()
        res_df = res_df.merge(feature_df, how='left', left_on='item_id', right_on='reference').set_index('index')
        res_df = res_df.drop('item_id', axis=1)
        res_df['reference'] = df['reference']
        return res_df.fillna(0).astype({'glob_inter_popularity': 'int'})


if __name__ == '__main__':
    import utils.menu as menu

    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    c = GlobalInteractionsPopularity(mode, cluster)
    
    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()
    
    print(c.read_feature())
