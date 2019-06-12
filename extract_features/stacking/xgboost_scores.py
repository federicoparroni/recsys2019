import pandas as pd
import numpy as np
import data
import os
import glob
from preprocess_utils.extract_scores import *
from extract_features.feature_base import FeatureBase
from preprocess_utils.last_clickout_indices import find
from pathlib import Path


class ScoresXGB(FeatureBase):

    """
        XGB scores
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'scoresxgb'
        super(ScoresXGB, self).__init__(name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        self.current_directory = Path(__file__).absolute().parent
        self.data_dir = self.current_directory.joinpath('..', '..', 'stacking', self.mode)

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self
                            .cluster)
        df = pd.concat([train, test])
        last_indices = find(df)

        # extract test scores
        self.train_dir = self.data_dir.joinpath('test')
        for file in glob.glob(str(self.train_dir) + '/xgboost*'):
            test_xgb = np.load(file)
            test_xgb = pd.DataFrame(test_xgb, columns=['index', 'item_recommendations', 'scores'])
            test_xgb = test_xgb.astype({'index': int})

        self.train_dir = self.data_dir.joinpath('train')
        for file in glob.glob(str(self.train_dir) + '/xgboost*'):
            train_xgb = np.load(file)
            train_xgb = pd.DataFrame(train_xgb, columns=['index', 'item_recommendations', 'scores'])
            train_xgb = train_xgb.astype({'index': int})

        xgb = pd.concat([train_xgb, test_xgb])

        # xgb_idx = list(xgb['index'])
        # print(f'Xgb indices are : {len(set(xgb_idx))}')
        # print(f'Last indices are : {len((last_indices))}')
        # common = set(xgb_idx) & set(last_indices)
        # print(f'In common : {len(common)}')
        xgb = xgb[xgb['index'].isin(last_indices)]

        xgb_idx = list(xgb['index'])

        t = assign_score(xgb, 'xgboost')
        t = t.sort_values(by='index')

        df['index'] = df.index.values
        df = df[['user_id', 'session_id','index']]
        df = pd.merge(t, df, how='left', on=['index'])
        return df[['user_id', 'session_id', 'item_id', 'score_xgboost']]



if __name__ == '__main__':

    c = ScoresXGB(mode='local', cluster='no_cluster')

    c.save_feature(overwrite_if_exists=True)

