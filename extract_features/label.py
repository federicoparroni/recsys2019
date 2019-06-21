import os
import sys
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
tqdm.pandas()
from preprocess_utils.last_clickout_indices import expand_impressions

def _reinsert_clickout(df):
    # take the row of the missing clickout
    clickout_rows_df = df[(df['action_type'] == 'clickout item') & df['reference'].isnull()]
    # check if it exsists
    if len(clickout_rows_df)>0:
        # retrieve from the full_df the clickout
        missing_click = data.full_df().loc[clickout_rows_df.index[0]]['reference']
        # reinsert the clickout on the df
        df.at[clickout_rows_df.index[0], 'reference']= missing_click
    return df

class ImpressionLabel(FeatureBase):

    """
    say for each impression of a clickout if it is the one clicked (1) or no 0
    | user_id | session_id | item_id | label
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_label'
        super(ImpressionLabel, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        if self.mode in ['small', 'local']:
            print('reinserting clickout')
            test = test.groupby(['session_id', 'user_id']).progress_apply(_reinsert_clickout)
        df = pd.concat([train, test])
        idxs_click = find_last_clickout_indices(df)
        df = df.loc[idxs_click][['user_id', 'session_id', 'reference', 'impressions']]
        df = expand_impressions(df)
        df['label'] = (df['item_id'] == df['reference'].astype('float'))*1
        df.drop(['index', 'reference'], axis=1, inplace=True)

        print(df)
        return df

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = ImpressionLabel(mode=mode, cluster=cluster)
    c.save_feature()
