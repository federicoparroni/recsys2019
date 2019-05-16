from tqdm.auto import tqdm
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
import numpy as np
import pandas as pd
import data
from extract_features.feature_base import FeatureBase
import sys
import os
sys.path.append(os.getcwd())


class SessionsImpressionsCountNumeric(FeatureBase):

    """
    counts the number of impressions at the moment of the clickout
    | index | session_impression_count
    """

    def __init__(self, mode='full', cluster='no_cluster'):
        name = 'session_impression_count'
        columns_to_onehot = []

        super().__init__(name=name, mode=mode, cluster=cluster,
                         columns_to_onehot=columns_to_onehot, save_index=False)

    def extract_feature(self):
        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        
        # find the last clickout rows
        last_clickout_idxs = find_last_clickout_indices(df)
        clickout_rows = df.loc[last_clickout_idxs, [
            'user_id', 'session_id', 'impressions']]
        clickout_rows['impressions_count'] = clickout_rows.impressions.str.split(
            '|').str.len()
        clickout_rows = clickout_rows.drop('impressions', axis=1)
        return clickout_rows

if __name__ == '__main__':
    from utils.menu import mode_selection
    mode = mode_selection()

    c = SessionsImpressionsCountNumeric(mode=mode)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

