import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
import numpy as np
from collections import Counter
from tqdm.auto import tqdm

class ImpressionsPopularity(FeatureBase):

    """
    Extracts the impressions popularity of the clickout rows.
    The count is unbiased by discarding 1 click for the reference clicked in the clickout.
    | index | impr_pop{i}
    impr_pop is positive integer, i is a number between 0-24
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'impressions_popularity'
        columns_to_onehot = []

        super().__init__(name=name, mode='full', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = data.full_df()

        # count the popularity
        #cnt = Counter(df[(df.action_type == 'clickout item') & (df.reference.str.isnumeric() == True)].reference.values.astype(int))
        pop_df = df[(df.action_type == 'clickout item') & (df.reference.str.isnumeric() == True)] \
                    [['reference','frequence']].astype('int').groupby('reference').sum()
        cnt = pop_df.to_dict()['frequence']
        
        # find the clickout rows
        clickout_rows = df[df.action_type == 'clickout item'][['reference','impressions']]
        clickout_rows = clickout_rows.fillna(-1).astype({'reference':'int'})
        clickout_rows['impressions'] = clickout_rows.apply(lambda x: list(map(int,x.impressions.split('|'))), axis=1)

        # build the resulting matrix
        matrix = np.zeros((clickout_rows.shape[0],25), dtype=int)
        
        i = 0
        for impr in tqdm(clickout_rows.impressions):
            for j,impr in enumerate(impr):
                ## OLD version
                #popularity = cnt[impr] if impr in cnt else 0
                #if popularity == row.reference:
                #    popularity -= 1

                ## NEW ! (decrease 1 to all references)
                popularity = cnt[impr]-1 if impr in cnt else 0
                matrix[i, j] = popularity
            i += 1
        
        # scale log and min-max
        min_pop = np.log((pop_df['frequence'] -1).clip(0).min() +1)
        max_pop = np.log((pop_df['frequence'] -1).clip(0).max() +1)
        
        matrix = (np.log(matrix + 1) - min_pop) / (max_pop - min_pop)

        # add the columns to the resulting dataframe
        for i in range(25):
            clickout_rows['impr_pop{}'.format(i)] = matrix[:, i]
        
        return clickout_rows.drop(['reference','impressions'], axis=1)


    def join_to(self, df, one_hot=False):
        """ Join this feature to the specified dataframe """
        feature_df = self.read_feature(one_hot=one_hot)
        feature_cols = feature_df.columns
        res_df = df.merge(feature_df, how='left', left_index=True, right_index=True)
        res_df[feature_cols] = res_df[feature_cols].fillna(0) #.astype('int')
        return res_df


if __name__ == '__main__':
    import utils.menu as menu

    c = ImpressionsPopularity()

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
