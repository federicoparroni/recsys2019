import sys
import os
sys.path.append(os.getcwd())

from extract_features.feature_base import FeatureBase
import data
import numpy as np
import pandas as pd
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from tqdm.auto import tqdm

class SessionLabel(FeatureBase):

    """
    Extracts the position of the reference inside the last clickout impressions.
    If the reference is not present in the next clickout impressions, the label will be 0.
    | index | user_id | session_id | label
    label is a number between 0-24
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'session_label'
        columns_to_onehot = []

        super().__init__(name=name, mode=mode, cluster=cluster, columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()

        df = pd.concat([data.train_df(self.mode), data.test_df(self.mode)])
        
        # find the last clickout rows
        idxs = find_last_clickout_indices(df)

        res_df = df[['user_id','session_id','reference','impressions']].loc[idxs]
        
        def add_label(row):
            impress = list(map(int, row['impressions'].split('|')))
            ref = row.reference
            if pd.isnull(ref):
                return np.NaN

            if ref in impress:
                return impress.index(ref)
            else:
                return 0
        
        res_df = res_df.astype({'reference':'float'})
        res_df['label'] = res_df.progress_apply(add_label, axis=1)
        res_df = res_df.dropna().astype({'label':'int'})

        return res_df.drop(['reference','impressions'], axis=1)



if __name__ == '__main__':
    import utils.menu as menu
    mode = menu.mode_selection()

    c = SessionLabel(mode=mode)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
