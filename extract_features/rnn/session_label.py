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

        super().__init__(name=name, mode=mode, cluster='no_cluster', columns_to_onehot=columns_to_onehot, save_index=True)


    def extract_feature(self):
        tqdm.pandas()
        """
        Train and test cannot be concatenated because there are some sessions that
        are splitted and they have a first part in train and the last in the test.
        In those cases, the label will be only one (since they will be treated as
        one session by 'find_last_clickout_indices' function) but they must have
        2 different labels (1 for the train half, 1 for the test half)
        """
        #df = pd.concat([data.train_df(self.mode), data.test_df(self.mode)])
        
        def get_label(df):
            """ Return a dataframe with: index | user_id | session_id | label """
            # find the last clickout rows
            idxs = find_last_clickout_indices(df)

            res_df = df[['user_id','session_id','reference','impressions']].loc[idxs]
            # remove the test sessions with reference NaN
            res_df = res_df.dropna(subset=['reference']).astype({'reference':'int'})
            # create impressions list
            res_df['impressions_list'] = res_df['impressions'].str.split('|').apply(lambda x: list(map(int,x)))
            res_df.drop('impressions', axis=1, inplace=True)

            label_series = np.zeros(res_df.shape[0], dtype='int8')
            # iterate over the rows
            k = 0
            for row in tqdm(zip(res_df['reference'], res_df['impressions_list'])):
                ref = row[0]
                impress = row[1]
                if ref in impress:
                    label_series[k] = impress.index(ref)
                k += 1
            # add the new column
            res_df['label'] = label_series

            return res_df.drop(['reference','impressions_list'], axis=1)
        
        # compute the labels for train and test
        label_train = get_label(data.train_df(self.mode))
        label_test = get_label(data.test_df(self.mode))
        return pd.concat([label_train, label_test])



if __name__ == '__main__':
    import utils.menu as menu
    mode = menu.mode_selection()

    c = SessionLabel(mode=mode)

    print('Creating {} for {} {}'.format(c.name, c.mode, c.cluster))
    c.save_feature()

    print(c.read_feature())
